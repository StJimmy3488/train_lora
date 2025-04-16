import os
# Force spawn method for multiprocessing
os.environ["PYTHONPATH"] = os.getcwd()  # Ensure imports work in subprocess
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_MULTIPROCESSING_SHARING_STRATEGY"] = "file_system"

import torch.multiprocessing as mp
# Environment variables must be set before any imports that might use CUDA
os.environ['PYTORCH_ENABLE_WORKER_BIN_IDENTIFICATION'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Optional, Dict
from enum import Enum
import logging
import sqlite3
from contextlib import contextmanager, asynccontextmanager
import uuid
from fastapi import BackgroundTasks
import shutil
from utils import resolve_image_path, process_images_and_captions, create_dataset, train_model
import asyncio
import os
import torch
import psutil
from functools import lru_cache
from torch.utils.data import get_worker_info
import weakref
import requests
import threading

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG logging level as suggested
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)
mp.set_sharing_strategy('file_system')

# Modify the global tracking dictionary to store processes instead of threads
running_processes: Dict[str, mp.Process] = {}

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

# Add this function for worker initialization
def worker_init_fn(worker_id):
    """Initialize worker process"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # Set worker-specific settings
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

def run_training_process(job_id: str, request_dict: dict):
    """The actual training process that runs in a separate process"""
    try:
        # Reconstruct request object from dict
        request = TrainingRequest(**request_dict)
        
        # Create new event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the training job
        loop.run_until_complete(run_training_job(job_id, request))
        
    except Exception as e:
        logger.exception("Training job failed")
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
                ("failed", str(e), job_id)
            )
            conn.commit()
    finally:
        loop.close()

@app.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """Start a new training job"""
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

        # Start the process without daemon flag
        process = mp.Process(
            target=run_training_process,
            args=(job_id, request.model_dump()),
        )
        process.start()
        
        # Store the process reference
        running_processes[job_id] = process
        
        # Store the PID
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET pid = ? WHERE job_id = ?",
                (process.pid, job_id)
            )
            conn.commit()

        return TrainingResponse(
            job_id=job_id,
            status="accepted",
            message="Training job started successfully"
        )

    except Exception as e:
        logger.exception("Failed to start training job")
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
                ("failed", str(e), job_id)
            )
            conn.commit()
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
        # Check if job exists
        job = conn.execute(
            "SELECT status, pid FROM jobs WHERE job_id = ?", 
            (job_id,)
        ).fetchone()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job['status'] in ('completed', 'failed'):
            raise HTTPException(status_code=400, detail="Job is not running")

        # Get the process
        process = running_processes.get(job_id)
        
        try:
            if process and process.is_alive():
                # Get the parent process and all its children
                try:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    
                    # Kill children first
                    for child in children:
                        try:
                            child.kill()  # Use kill() instead of terminate()
                        except psutil.NoSuchProcess:
                            pass
                    
                    # Kill parent process
                    parent.kill()  # Use kill() instead of terminate()
                    
                    # Clean up process from tracking
                    process.join(timeout=1)  # Give it a second to clean up
                    running_processes.pop(job_id, None)
                    
                except psutil.NoSuchProcess:
                    logger.warning(f"Process {process.pid} not found")
                except Exception as e:
                    logger.error(f"Error killing process tree: {e}")
                    
            elif job['pid']:
                # Fallback to killing by PID
                try:
                    os.kill(job['pid'], 9)  # SIGKILL
                except ProcessLookupError:
                    logger.warning(f"PID {job['pid']} not found")
                except Exception as e:
                    logger.error(f"Error killing process by PID: {e}")
                    
        except Exception as e:
            logger.error(f"Error during process termination: {e}")
            
        finally:
            # Always update the job status
            conn.execute(
                "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
                ("failed", "Job was killed by user", job_id)
            )
            conn.commit()
            
            # Clean up any temporary files
            try:
                dataset_folder = f"tmp_datasets/{job_id}"
                if os.path.exists(dataset_folder):
                    shutil.rmtree(dataset_folder, ignore_errors=True)
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
        
        return {"message": f"Job {job_id} terminated"}

async def run_training_job(job_id: str, request: TrainingRequest):
    """Run the training job in the background"""
    try:
        # Helper function to check if job should stop
        def should_stop():
            with get_db() as conn:
                status = conn.execute(
                    "SELECT status FROM jobs WHERE job_id = ?",
                    (job_id,)
                ).fetchone()
                return status and status['status'] == 'stopping'

        # Update job status - downloading images
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ? WHERE job_id = ?",
                ("downloading_images", 0.1, job_id)
            )
            conn.commit()

        # Check for stop signal throughout the process
        if should_stop():
            return

        # Rest of your existing code, but add checks for should_stop() at key points
        # For example:
        
        images = []
        for image_url in request.images:
            if should_stop():
                return
            local_path = resolve_image_path(image_url)
            images.append(local_path)

        if should_stop():
            return

        # Continue with the rest of your existing code, adding should_stop() checks
        # at appropriate intervals...

        # Process images and generate captions
        captions = process_images_and_captions(images, request.concept_sentence)

        # Update status - creating dataset
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ? WHERE job_id = ?",
                ("creating_dataset", 0.4, job_id)
            )
            conn.commit()

        # Create dataset
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
        folder_url = await train_model(
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
    finally:
        # Clean up process reference
        running_processes.pop(job_id, None)
        # Clean up any temporary files
        try:
            if 'dataset_folder' in locals():
                shutil.rmtree(dataset_folder, ignore_errors=True)
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

def resolve_image_path(image_item):
    """Handle multiple possible 'image' input formats"""
    if not image_item:
        raise HTTPException(
            status_code=400,
            detail="Empty image path or URL provided"
        )

    # If it's a dict with 'name' key
    if isinstance(image_item, dict) and "name" in image_item:
        if not image_item["name"]:
            raise HTTPException(
                status_code=400,
                detail="Empty image name in dictionary"
            )
        return image_item["name"]

    # If it's a string (URL or path)
    if isinstance(image_item, str):
        # Handle URLs
        if image_item.lower().startswith(("http://", "https://")):
            os.makedirs("tmp_downloads", exist_ok=True)
            local_name = os.path.join("tmp_downloads", f"{uuid.uuid4()}.png")
            
            try:
                response = requests.get(image_item, stream=True)
                response.raise_for_status()
                with open(local_name, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return local_name
            except Exception as ex:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image from URL {image_item}: {str(ex)}"
                )

        # Handle local paths
        if os.path.exists(image_item):
            return image_item
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Image file not found: {image_item}"
            )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported image format: {type(image_item)}"
    )

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)
