from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Optional
from enum import Enum
import logging
import sqlite3
from contextlib import contextmanager, asynccontextmanager
import uuid
from fastapi import BackgroundTasks
import shutil
from utils import resolve_image_path, process_images_and_captions, create_dataset, train_model
import asyncio
from multiprocessing import Process
import os
import multiprocessing
import torch
import psutil
from functools import lru_cache

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG logging level as suggested
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)


class ModelType(str, Enum):
    dev = "dev"
    schnell = "schnell"

class TrainingRequest(BaseModel):
    images: List[str]
    lora_name: str
    concept_sentence: Optional[str] = None
    steps: int = 1000
    lr: float = 4e-4
    rank: int = 16
    model_type: ModelType = ModelType.dev
    low_vram: bool = True
    sample_prompts: Optional[List[str]] = None
    advanced_options: Optional[str] = None

    @field_validator('images')
    @classmethod
    def validate_images(cls, v: List[str]) -> List[str]:
        for url in v:
            if not (url.startswith('http://') or url.startswith('https://')):
                raise ValueError(f"Invalid image URL: {url}. Must start with http:// or https://")
        return v

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class TrainingStatus(BaseModel):
    status: str
    progress: Optional[float] = None
    folder_url: Optional[str] = None
    error: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    logger.info("Database initialized")
    yield
    # Shutdown
    pass

app = FastAPI(
    title="FLUX LoRA Trainer API",
    description="Train FLUX LoRA models with your images",
    version="1.0.0",
    lifespan=lifespan
)

# Replace in-memory dictionary with SQLite
@contextmanager
def get_db():
    conn = sqlite3.connect('jobs.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        # First create the table if it doesn't exist (without pid column to maintain compatibility)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                progress REAL,
                folder_url TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Then add the pid column if it doesn't exist
        try:
            conn.execute('ALTER TABLE jobs ADD COLUMN pid INTEGER')
            logger.info("Added pid column to jobs table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                logger.debug("pid column already exists")
            else:
                logger.error(f"Error adding pid column: {e}")

def run_training_process(job_id: str, request_data: dict):
    """Run training in a separate process"""
    # Set up logging in the new process
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    try:
        # Set environment variables to control torch multiprocessing
        os.environ['PYTORCH_ENABLE_WORKER_BIN_IDENTIFICATION'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # Initialize torch multiprocessing settings
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        mp.set_sharing_strategy('file_system')
        
        # Initialize CUDA if available
        if torch.cuda.is_available():
            logger.info("Initializing CUDA in child process")
            torch.cuda.init()
            logger.info(f"CUDA initialized. Device: {torch.cuda.get_device_name(0)}")

        # Convert dict back to TrainingRequest
        request = TrainingRequest(**request_data)
        
        # Create a new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(run_training_job(job_id, request))
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Training process failed: {e}", exc_info=True)
        try:
            with get_db() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
                    ("failed", str(e), job_id)
                )
                conn.commit()
        except Exception as db_error:
            logger.error(f"Failed to update job status: {db_error}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new training job"""
    # Add request validation caching
    @lru_cache(maxsize=100)
    def validate_image_url(url: str) -> bool:
        return url.startswith(('http://', 'https://'))
    
    # Validate all URLs in parallel
    validation_tasks = [validate_image_url(url) for url in request.images]
    if not all(validation_tasks):
        raise HTTPException(status_code=400, detail="Invalid image URLs")

    # Check if there's already a running job
    with get_db() as conn:
        running_job = conn.execute(
            "SELECT job_id FROM jobs WHERE status NOT IN ('completed', 'failed')"
        ).fetchone()
        
        if running_job:
            raise HTTPException(
                status_code=409, 
                detail="Another training job is already in progress"
            )
    
    job_id = str(uuid.uuid4())
    try:
        # Store initial job status
        with get_db() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status, progress) VALUES (?, ?, ?)",
                (job_id, "initializing", 0.0)
            )
            conn.commit()

        # Create a new context for the process
        ctx = multiprocessing.get_context('spawn')
        
        # Start training in a separate process with specific context
        process = ctx.Process(
            target=run_training_process,
            args=(job_id, request.model_dump()),
            name=f"training-{job_id}"
        )
        
        # Start the process
        process.start()
        
        logger.info(f"Started training process with PID: {process.pid}")
        
        try:
            with get_db() as conn:
                conn.execute(
                    "UPDATE jobs SET pid = ? WHERE job_id = ?",
                    (process.pid, job_id)
                )
                conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not update process ID in database: {e}")
        
        return TrainingResponse(
            job_id=job_id,
            status="accepted",
            message="Training job started successfully"
        )
    except Exception as e:
        logger.exception("Failed to start training job")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get the status of a training job"""
    with get_db() as conn:
        job = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", 
            (job_id,)
        ).fetchone()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return TrainingStatus(
            status=job['status'],
            progress=job['progress'],
            folder_url=job['folder_url'],
            error=job['error']
        )

@app.get("/current-job")
async def get_current_job():
    """Get the ID of the currently running job, if any"""
    with get_db() as conn:
        running_job = conn.execute(
            "SELECT job_id, status, progress FROM jobs WHERE status NOT IN ('completed', 'failed')"
        ).fetchone()
        
        if running_job:
            return {
                "job_id": running_job['job_id'],
                "status": running_job['status'],
                "progress": running_job['progress']
            }
        return {"message": "No job currently running"}

@app.post("/kill/{job_id}")
async def kill_job(job_id: str):
    """Kill a running job"""
    with get_db() as conn:
        # Check if job exists and get its PID
        job = conn.execute(
            "SELECT status, pid FROM jobs WHERE job_id = ?", 
            (job_id,)
        ).fetchone()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job['status'] in ('completed', 'failed'):
            raise HTTPException(status_code=400, detail="Job is not running")
        
        # Try to terminate the process
        if job['pid']:
            try:
                process = psutil.Process(job['pid'])
                process.terminate()  # or process.kill() for force kill
                
                # Wait for process to terminate
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    process.kill()  # Force kill if graceful termination fails
                    
            except psutil.NoSuchProcess:
                logger.warning(f"Process {job['pid']} not found")
            except Exception as e:
                logger.error(f"Error killing process: {e}")
        
        # Update job status to failed
        conn.execute(
            "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
            ("failed", "Job was killed by user", job_id)
        )
        conn.commit()
        
        return {"message": f"Job {job_id} terminated"}

async def run_training_job(job_id: str, request: TrainingRequest):
    """Run the training job in the background"""
    try:
        # Set environment variables for single process operation
        os.environ['PYTORCH_ENABLE_WORKER_BIN_IDENTIFICATION'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        
        # Force single-threaded operation
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        
        # Update job status - downloading images
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ? WHERE job_id = ?",
                ("downloading_images", 0.1, job_id)
            )
            conn.commit()
        
        # Download and process images
        images = []
        for image_url in request.images:
            local_path = await resolve_image_path(image_url)
            images.append(local_path)

        # Update status - processing images
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ? WHERE job_id = ?",
                ("processing_images", 0.2, job_id)
            )
            conn.commit()

        # Process images and generate captions
        captions = process_images_and_captions(images, request.concept_sentence)

        # Update status - creating dataset
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ? WHERE job_id = ?",
                ("creating_dataset", 0.4, job_id)
            )
            conn.commit()

        # Create dataset - now with await
        dataset_folder = await create_dataset(images, captions)

        # Update status - training
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ? WHERE job_id = ?",
                ("training", 0.5, job_id)
            )
            conn.commit()

        # Convert sample_prompts from list to comma-separated string if provided
        sample_prompts_str = None
        if request.sample_prompts:
            sample_prompts_str = ",".join(request.sample_prompts)

        # Train model
        folder_url =  train_model(
            dataset_folder=dataset_folder,
            lora_name=request.lora_name,
            concept_sentence=request.concept_sentence,
            steps=request.steps,
            lr=request.lr,
            rank=request.rank,
            model_type=request.model_type,
            low_vram=request.low_vram,
            sample_prompts=sample_prompts_str,
            advanced_options=request.advanced_options
        )

        # Update final status
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ?, folder_url = ? WHERE job_id = ?",
                ("completed", 1.0, folder_url, job_id)
            )
            conn.commit()

        logger.info(f"Training completed successfully. Model available at: {folder_url}")

    except Exception as e:
        logger.exception("Training job failed")
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
                ("failed", str(e), job_id)
            )
            conn.commit()
        # Cleanup any temporary files if needed
        try:
            if 'dataset_folder' in locals():
                shutil.rmtree(dataset_folder, ignore_errors=True)
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")


# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    try:
        # Set environment variables for the main process
        os.environ['PYTORCH_ENABLE_WORKER_BIN_IDENTIFICATION'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # Initialize multiprocessing settings
        multiprocessing.set_start_method('spawn')
        
        # Initialize torch multiprocessing settings
        import torch.multiprocessing as mp
        mp.set_sharing_strategy('file_system')
        
        # Initialize CUDA in the main process
        if torch.cuda.is_available():
            torch.cuda.init()
            logger.info(f"CUDA initialized in main process. Device: {torch.cuda.get_device_name(0)}")
        
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)
    except RuntimeError as e:
        if "context has already been set" in str(e):
            # Ignore if context is already set
            pass
        else:
            raise 