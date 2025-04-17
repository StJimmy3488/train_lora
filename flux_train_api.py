import os
import sys
import uuid
import shutil
import json
import yaml
import logging
import time 
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from enum import Enum
import sqlite3
from contextlib import contextmanager, asynccontextmanager

from PIL import Image
from dotenv import load_dotenv
import requests  
import torch
import boto3
from botocore.exceptions import NoCredentialsError
from slugify import slugify

from transformers import AutoProcessor, AutoModelForCausalLM
from botocore.config import Config

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "ai-toolkit")
from toolkit.job import get_job

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout
        logging.FileHandler('training.log')  # Also log to file
    ]
)


import boto3
from botocore.config import Config
import os
import gc
import subprocess


def get_s3_client():
    """Create S3-compatible client for Cloudflare R2"""
    s3_endpoint = os.getenv("S3_ENDPOINT")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_REGION")

    if not s3_endpoint or not aws_access_key_id or not aws_secret_access_key:
        raise RuntimeError("Missing S3 credentials. Check .env file.")

    config = Config(
        signature_version="s3v4",  # ✅ Explicitly enforce Signature v4
        retries={"max_attempts": 3, "mode": "standard"}
    )

    return boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
        config=config
    )


def upload_directory_to_s3(local_dir, bucket_name, s3_prefix):
    """Uploads a directory to S3 while maintaining folder structure"""
    logger.info("Starting S3 upload from '%s' to bucket '%s' with prefix '%s'",
                local_dir, bucket_name, s3_prefix)
    
    if not os.path.exists(local_dir):
        logger.error("Local directory '%s' does not exist", local_dir)
        return False
        
    try:
        s3 = get_s3_client()
        
        # Count files first
        total_files = sum([len(files) for _, _, files in os.walk(local_dir)])
        logger.info("Found %d files to upload", total_files)
        
        uploaded_files = 0
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                try:
                    logger.debug("Uploading file %d/%d: '%s' to '%s'",
                               uploaded_files + 1, total_files, local_path, s3_key)
                    s3.upload_file(local_path, bucket_name, s3_key)
                    uploaded_files += 1
                except Exception as e:
                    logger.error("Failed to upload '%s': %s", local_path, e)
                    return False
                
        logger.info("Successfully uploaded %d/%d files", uploaded_files, total_files)
        return uploaded_files > 0  # Return True only if we uploaded at least one file
        
    except NoCredentialsError:
        logger.error("AWS credentials not available")
        return False
    except Exception as e:
        logger.exception("Error during S3 upload: %s", e)
        return False
    finally:
        try:
            logger.debug("Cleaning up local directory '%s'", local_dir)
            shutil.rmtree(local_dir, ignore_errors=True)
        except Exception as e:
            logger.error("Error cleaning up local directory: %s", e)


async def resolve_image_path(image_item):
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

def process_images_and_captions(images, concept_sentence=None):
    logger.debug("Processing images for captioning. Number of images: %d", len(images))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = None
    processor = None
    captions = []
    local_image_paths = []

    try:
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(device)

        processor = AutoProcessor.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn",
            trust_remote_code=True
        )

        # Resolve URLs to local paths synchronously
        for image_item in images:
            local_path = requests.get(image_item, stream=True)
            local_path.raise_for_status()

            os.makedirs("tmp_downloads", exist_ok=True)
            file_path = os.path.join("tmp_downloads", f"{uuid.uuid4()}.png")

            with open(file_path, "wb") as f:
                for chunk in local_path.iter_content(chunk_size=8192):
                    f.write(chunk)

            local_image_paths.append(file_path)

        for image_path in local_image_paths:
            try:
                with Image.open(image_path) as raw:
                    image = raw.convert("RGB")

                prompt = "<DETAILED_CAPTION>"
                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(device, torch_dtype)

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )

                generated_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=False
                )[0]

                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task=prompt,
                    image_size=(image.width, image.height)
                )

                caption = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
                if concept_sentence:
                    caption = f"{caption} [trigger]"

                captions.append(caption)
                logger.debug("Caption generated for %s: %s", image_path, caption)
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                raise

    finally:
        # Cleanup downloaded images
        for path in local_image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed temporary image file: {path}")
            except Exception as e:
                logger.error(f"Error removing temporary file {path}: {e}")

        # Aggressive GPU memory cleanup
        if torch.cuda.is_available():
            if model is not None:
                model.cpu()
                del model
            if processor is not None:
                del processor
            torch.cuda.empty_cache()
            gc.collect()

    logger.debug("Generated %d captions", len(captions))
    if len(captions) == 0:
        raise RuntimeError("No captions were generated from the images")

    return captions

async def create_dataset(images, captions):
    """Create temporary dataset from images and captions"""
    if len(images) != len(captions):
        raise RuntimeError(f"Number of images ({len(images)}) does not match number of captions ({len(captions)})")
        
    destination_folder = f"tmp_datasets/{uuid.uuid4()}"
    logger.info("Creating a dataset in folder: %s", destination_folder)
    os.makedirs(destination_folder, exist_ok=True)

    jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
    processed_count = 0
    
    with open(jsonl_file_path, "a") as jsonl_file:
        for image_item, caption in zip(images, captions):
            try:
                local_path = await resolve_image_path(image_item)
                logger.debug("Copying %s to dataset folder %s", local_path, destination_folder)
                new_image_path = shutil.copy(local_path, destination_folder)
                file_name = os.path.basename(new_image_path)
                data = {"file_name": file_name, "prompt": caption}
                jsonl_file.write(json.dumps(data) + "\n")
                logger.debug("Wrote to metadata.jsonl: %s", data)
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process image {image_item}: {e}")
                raise

    logger.info(f"Successfully processed {processed_count} images to dataset")
    if processed_count == 0:
        raise RuntimeError("No images were successfully processed for the dataset")

    return destination_folder

async def train_model(
    dataset_folder,
    lora_name,
    concept_sentence=None,
    steps=1000,
    lr=4e-4,
    rank=16,
    model_type="dev",
    low_vram=True,
    sample_prompts=None,
    advanced_options=None
):
    """Train the model and store exclusively in S3, returning a folder URL."""
    try:
        slugged_lora_name = slugify(lora_name)
        logger.info("Training LoRA model. Name: %s, Slug: %s", lora_name, slugged_lora_name)

        # Load default config
        logger.debug("Loading default config file: config/examples/train_lora_flux_24gb.yaml")
        with open("config/examples/train_lora_flux_24gb.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Clear memory before starting
        clear_gpu_memory()
        
        # Update configuration to be more memory-efficient
        config["config"]["process"][0].update({
            "model": {
                "low_vram": True,  # Force low VRAM mode
                "name_or_path": "black-forest-labs/FLUX.1-schnell" if model_type == "schnell" else "black-forest-labs/FLUX.1",
                "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter" if model_type == "schnell" else None,
                "gradient_checkpointing": True,  # Enable gradient checkpointing
                "enable_xformers_memory_efficient_attention": True,  # Enable memory efficient attention
                "model_cpu_offload": True,  # Enable CPU offloading
            },
            "train": {
                "skip_first_sample": True,
                "steps": int(steps),
                "lr": float(lr),
                "batch_size": 1,  # Minimum batch size
                "gradient_accumulation_steps": 4,
                "mixed_precision": "fp16",  # Use mixed precision training
                "full_fp16": True,  # Use full fp16 training
            },
            "network": {
                "linear": int(rank),
                "linear_alpha": int(rank)
            },
            "datasets": [{"folder_path": dataset_folder}],
            "save": {
                "output_dir": f"tmp_models/{slugged_lora_name}",
                "push_to_hub": False  # Disable Hugging Face push
            }
        })

        if concept_sentence:
            logger.debug("Setting concept_sentence (trigger_word) to '%s'.", concept_sentence)
            config["config"]["process"][0]["trigger_word"] = concept_sentence

        if sample_prompts:
            logger.debug("Sample prompts provided. Will enable sampling.")
            config["config"]["process"][0]["train"]["disable_sampling"] = False
            config["config"]["process"][0]["sample"].update({
                "sample_every": steps,
                "sample_steps": 28 if model_type == "dev" else 4,
                "prompts": sample_prompts
            })
        else:
            logger.debug("No sample prompts provided. Disabling sampling.")
            config["config"]["process"][0]["train"]["disable_sampling"] = True

        if advanced_options:
            logger.debug("Merging advanced_options YAML into config.")
            config["config"]["process"][0] = recursive_update(
                config["config"]["process"][0],
                yaml.safe_load(advanced_options)
            )

        # Save config
        config_path = f"tmp_configs/{uuid.uuid4()}-{slugged_lora_name}.yaml"
        logger.debug("Saving updated config to: %s", config_path)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Run training
        logger.info("Retrieving job with config path: %s", config_path)
        job = get_job(config_path, slugged_lora_name)

        logger.debug("job object => %s", job)
        if job is None:
            raise RuntimeError(f"get_job() returned None for config path: {config_path}.")

        try:
            job_start_time = time.time()
            logger.info("Running job...")
            job.run()
            job_runtime = time.time() - job_start_time
            logger.info(f"Job run completed in {job_runtime:.2f} seconds.")
        finally:
            # Ensure cleanup happens even if job.run() fails
            logger.info("Cleaning up job resources...")
            if hasattr(job, 'cleanup') and callable(job.cleanup):
                job.cleanup()
            
            # Manually clean up any remaining references
            if hasattr(job, 'model'):
                if hasattr(job.model, 'to'):
                    job.model.to('cpu')
                del job.model
            
            # Force garbage collection
            clear_gpu_memory()
            
            # Clear the job reference
            del job

        # Check S3 configuration early
        bucket_name = os.environ.get("S3_BUCKET")
        if not bucket_name:
            raise RuntimeError("S3_BUCKET environment variable is not set")
        
        s3_domain = os.getenv("S3_DOMAIN")
        if not s3_domain:
            raise RuntimeError("S3_DOMAIN environment variable is not set")

        # After training, verify output directory
        local_model_dir = f"output/{slugged_lora_name}"
        logger.info("Checking for model output directory: %s", local_model_dir)
        
        if not os.path.exists(local_model_dir):
            # Log directory contents to help debug
            output_dir = "output"
            if os.path.exists(output_dir):
                logger.error("Contents of output directory:")
                for item in os.listdir(output_dir):
                    logger.error("  - %s", item)
            else:
                logger.error("Output directory does not exist")
            raise RuntimeError(f"Model output directory '{local_model_dir}' was not created during training")

        # Log directory contents before upload
        logger.info("Contents of model directory before upload:")
        for root, dirs, files in os.walk(local_model_dir):
            for file in files:
                logger.info("  - %s", os.path.join(root, file))

        s3_prefix = f"loras/flux/{slugged_lora_name}"
        logger.info("Uploading trained model to S3: bucket=%s, prefix=%s", bucket_name, s3_prefix)
        
        if upload_directory_to_s3(local_model_dir, bucket_name, s3_prefix):
            s3_folder_url = f"{s3_domain}/{s3_prefix}/"
            logger.info("Model folder successfully uploaded to: %s", s3_folder_url)
            return s3_folder_url
        else:
            raise RuntimeError("Failed to upload model to S3")

    except Exception as e:
        logger.exception("Error during model training or upload")
        clear_gpu_memory()
        raise

    finally:
        # Cleanup code
        try:
            if os.path.exists(config_path):
                os.remove(config_path)
                logger.debug("Removed config file: %s", config_path)
            if os.path.exists(dataset_folder):
                shutil.rmtree(dataset_folder, ignore_errors=True)
                logger.debug("Removed dataset folder: %s", dataset_folder)
        except Exception as e:
            logger.error("Error during cleanup: %s", e)
        
        clear_gpu_memory()


def recursive_update(d, u):
    """Recursively update nested dictionaries"""
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

async def train_lora(
    images,
    lora_name,
    concept_sentence=None,
    steps=1000,
    lr=4e-4,
    rank=16,
    model_type="dev",
    low_vram=True,
    sample_prompts=None,
    advanced_options=None
):
    """Main training workflow with improved logging and error handling."""
    logger.info("Initiating train_lora with the following parameters: "
                "lora_name=%s, concept_sentence=%s, steps=%s, lr=%s, rank=%s, model_type=%s, low_vram=%s, sample_prompts=%s",
                lora_name, concept_sentence, steps, lr, rank, model_type, low_vram, sample_prompts)

    try:
        logger.info("1. Process images and captions.")
        captions = process_images_and_captions(images, concept_sentence)

        logger.info("2. Create dataset from images and captions.")
        dataset_folder = await create_dataset(images, captions)

        logger.info("3. Train the model using train_model function.")
        folder_url = await train_model(
            dataset_folder,
            lora_name,
            concept_sentence,
            steps,
            lr,
            rank,
            model_type,
            low_vram,
            sample_prompts,
            advanced_options
        )
        logger.info("Training complete. Folder URL: %s", folder_url)
        return {"status": "success", "folder_url": folder_url}
    except RuntimeError as e_rt: # Catch RuntimeErrors specifically from train_model or other critical functions
        logger.error(f"RuntimeError in train_lora (likely training or S3 upload failure): {e_rt}")
        return {"status": "error", "message": str(e_rt)} # Return specific RuntimeError message
    except Exception as e: # Catch any other unexpected exceptions in train_lora itself
        logger.exception("Unexpected error in train_lora: %s", e) # Log full exception with traceback
        return {"status": "error", "message": "An unexpected error occurred during training. Check server logs for details."} # More generic message for client
    finally:
        # Cleanup temporary directories - moved to train_model's finally block to ensure cleanup even if train_model fails.
        logger.debug("Final cleanup (check if tmp dirs are removed by train_model's finally).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.7)  # Reduce to 70% to leave more headroom
        torch.cuda.empty_cache()
    init_db()
    yield
    # Shutdown
    clear_gpu_memory()

app = FastAPI(title="FLUX LoRA Trainer API", lifespan=lifespan)

# Define enums and models
class ModelType(str, Enum):
    dev = "dev"
    schnell = "schnell"

class TrainingRequest(BaseModel):
    images: List[str] = Field(..., description="List of image URLs to train on")
    lora_name: str = Field(..., description="Name for the LoRA model")
    concept_sentence: Optional[str] = Field(None, description="Optional trigger word/sentence")
    steps: int = Field(default=1000, ge=100, le=10000, description="Number of training steps")
    lr: float = Field(default=4e-4, gt=0, le=1, description="Learning rate")
    rank: int = Field(default=16, ge=1, le=128, description="LoRA rank")
    model_type: ModelType = Field(default=ModelType.dev, description="Model type (dev or schnell)")
    low_vram: bool = Field(default=True, description="Enable low VRAM mode")
    sample_prompts: Optional[List[str]] = Field(None, description="Optional sample prompts for testing")
    advanced_options: Optional[str] = Field(None, description="Advanced YAML configuration")

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: Optional[str] = None
    status: str
    progress: float
    folder_url: Optional[str] = None
    error: Optional[str] = None

def get_subprocess_env():
    """Get environment variables for subprocess"""
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    return env

def run_training_in_subprocess(request_dict, job_id):
    """Run the training in a separate Python process"""
    try:
        logger.info(f"Starting training subprocess for job {job_id}")
        request_json = json.dumps(request_dict)
        
        logger.debug("Subprocess environment configuration:")
        env = get_subprocess_env()
        for key in ['PYTORCH_CUDA_ALLOC_CONF', 'CUDA_VISIBLE_DEVICES']:
            logger.debug(f"  {key}: {env.get(key, 'not set')}")
        
        logger.info("Launching subprocess with train_subprocess.py")
        process = subprocess.Popen(
            [sys.executable, "train_subprocess.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Use a queue to capture the final result
        import queue
        result_queue = queue.Queue()
        
        def log_output(pipe, level, is_stdout=False):
            final_output = []
            for line in iter(pipe.readline, ''):
                if level == logging.INFO:
                    if line.startswith('{"status":'):
                        # This is our JSON result
                        result_queue.put(line.strip())
                    else:
                        logger.info(f"Subprocess: {line.strip()}")
                        if is_stdout:
                            final_output.append(line.strip())
                else:
                    logger.error(f"Subprocess error: {line.strip()}")
            if is_stdout and not result_queue.qsize():
                # If no JSON result was found, store the full output
                result_queue.put('\n'.join(final_output))
        
        import threading
        stdout_thread = threading.Thread(target=log_output, args=(process.stdout, logging.INFO, True))
        stderr_thread = threading.Thread(target=log_output, args=(process.stderr, logging.ERROR, False))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        logger.info("Sending request data to subprocess")
        process.stdin.write(request_json + "\n")
        process.stdin.flush()
        
        # Wait for process to complete
        returncode = process.wait(timeout=7200)
        stdout_thread.join(timeout=5)  # Add timeout to thread joining
        stderr_thread.join(timeout=5)
        
        if returncode != 0:
            logger.error(f"Subprocess failed with return code {returncode}")
            return {"status": "error", "message": "Training process failed"}
        
        try:
            # Get the result from the queue
            final_output = result_queue.get(timeout=5)
            logger.debug(f"Raw subprocess output: {final_output}")
            
            try:
                result = json.loads(final_output)
                logger.info(f"Training result: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse subprocess output: {e}\nOutput was: {final_output}")
                return {"status": "error", "message": "Training process output was not in the expected format"}
                
        except queue.Empty:
            logger.error("No output received from subprocess")
            return {"status": "error", "message": "No output received from training process"}
            
    except subprocess.TimeoutExpired:
        logger.error("Training subprocess timed out after 2 hours")
        process.kill()
        return {"status": "error", "message": "Training timed out after 2 hours"}
    except Exception as e:
        logger.exception("Error running training subprocess")
        if 'process' in locals() and process.poll() is None:
            process.kill()
        return {"status": "error", "message": str(e)}

def run_training_job(job_id: str, request: TrainingRequest):
    """Background task to run the training job"""
    try:
        # Update job status to processing
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ? WHERE job_id = ?",
                ("processing", 0.1, job_id)
            )
            conn.commit()

        # Convert request to dict for subprocess
        request_dict = request.model_dump()  # Use model_dump instead of dict
        
        # Run training in subprocess
        result = run_training_in_subprocess(request_dict, job_id)

        if result["status"] == "success":
            # Update final status
            with get_db() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, progress = ?, folder_url = ? WHERE job_id = ?",
                    ("completed", 1.0, result["folder_url"], job_id)
                )
                conn.commit()
        else:
            # Update error status
            with get_db() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
                    ("failed", result["message"], job_id)
                )
                conn.commit()

    except Exception as e:
        logger.exception("Training job failed")
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
                ("failed", str(e), job_id)
            )
            conn.commit()

@app.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
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

    # Create new job ID
    job_id = str(uuid.uuid4())

    # Initialize job in database
    with get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, progress) VALUES (?, ?, ?)",
            (job_id, "initializing", 0.0)
        )
        conn.commit()

    # Add the task to background tasks
    background_tasks.add_task(run_training_job, job_id, request)

    return TrainingResponse(
        job_id=job_id,
        status="accepted",
        message="Training job started successfully"
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a training job"""
    with get_db() as conn:
        result = conn.execute(
            "SELECT job_id, status, progress, folder_url, error FROM jobs WHERE job_id = ?",
            (job_id,)
        ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=result[0],
        status=result[1],
        progress=result[2],
        folder_url=result[3],
        error=result[4]
    )

@app.get("/current", response_model=Optional[JobStatus])
async def get_current_job():
    """Get information about the currently running job if any"""
    with get_db() as conn:
        result = conn.execute(
            "SELECT job_id, status, progress, folder_url, error FROM jobs "
            "WHERE status NOT IN ('completed', 'failed') "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

    if not result:
        return None

    return JobStatus(
        job_id=result[0],
        status=result[1],
        progress=result[2],
        folder_url=result[3],
        error=result[4]
    )

@app.post("/kill")
async def kill_current_job():
    """Kill the currently running job"""
    with get_db() as conn:
        # Get current running job
        current_job = conn.execute(
            "SELECT job_id FROM jobs "
            "WHERE status NOT IN ('completed', 'failed') "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        
        if not current_job:
            raise HTTPException(
                status_code=404, 
                detail="No running job found"
            )
        
        job_id = current_job[0]
        
        # Update job status to failed
        conn.execute(
            "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
            ("failed", "Job was killed by user", job_id)
        )
        conn.commit()
        
        return {"message": "Current job terminated"}

# Database helper
@contextmanager
def get_db():
    """Database connection context manager"""
    conn = sqlite3.connect("jobs.db")
    try:
        yield conn
    finally:
        conn.close()

# Initialize database
def init_db():
    """Initialize the database with required tables"""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0,
                folder_url TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

@contextmanager
def device_context(device="cuda"):
    try:
        yield device
    finally:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        # Clear PyTorch's CUDA memory cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Optional: wait a moment for memory to be freed
        time.sleep(1)
        
        # Log memory stats
        logger.debug(
            "GPU Memory: Allocated: %.1f GB, Reserved: %.1f GB, Max: %.1f GB",
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
            torch.cuda.max_memory_allocated() / 1e9
        )

@app.get("/test_logging")
async def test_logging():
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    return {"message": "Logging test completed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)