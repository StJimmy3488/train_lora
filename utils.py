import os
import uuid
import requests
import logging
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from fastapi import  HTTPException
import shutil
import json
from s3_utils import upload_directory_to_s3
import time
import yaml
from slugify import slugify
from toolkit.job import get_job
import asyncio
from contextlib import nullcontext
from io import BufferedWriter
from s3_utils import get_s3_client

logger = logging.getLogger(__name__)

# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
# print(f"GPU device: {torch.cuda.get_device_name(0)}")
# print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")


def process_images_and_captions(images, concept_sentence=None):
    """Process uploaded images and generate captions if needed"""
    logger.debug("Processing images for captioning. Number of images: %d", len(images))

    if len(images) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least 2 images"
        )
    elif len(images) > 150:
        raise HTTPException(
            status_code=400,
            detail="Maximum 150 images allowed"
        )

    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    logger.info(f"Using device: {device} with dtype: {torch_dtype}")

    # Add memory tracking
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
        logger.debug(f"Initial GPU memory: {initial_memory / 1024**2:.2f}MB")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device
        ).to(device)

        processor = AutoProcessor.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn",
            trust_remote_code=True
        )

        # Increase batch size if GPU memory allows
        batch_size = 8  # Increased from 4
        
        # Pre-load all images before processing
        all_pil_images = []
        for image_path in images:
            if isinstance(image_path, str):
                # Add basic image validation/optimization
                image = Image.open(image_path).convert("RGB")
                # Resize large images to reasonable dimensions
                if max(image.size) > 2048:
                    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                all_pil_images.append(image)
        
        # Process in larger batches
        captions = []
        for i in range(0, len(all_pil_images), batch_size):
            batch_pil_images = all_pil_images[i:i + batch_size]
            
            # Process batch
            prompt = "<DETAILED_CAPTION>"
            inputs = processor(
                text=[prompt] * len(batch_pil_images),
                images=batch_pil_images,
                return_tensors="pt"
            ).to(device, torch_dtype)

            with torch.amp.autocast('cuda') if device == "cuda" else nullcontext():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )

                generated_texts = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

            # Process generated texts
            for idx, (generated_text, pil_image) in enumerate(zip(generated_texts, batch_pil_images)):
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task=prompt,
                    image_size=(pil_image.width, pil_image.height)
                )

                caption = parsed_answer["<DETAILED_CAPTION>"].replace(
                    "The image shows ", ""
                )
                if concept_sentence:
                    caption = f"{caption} [trigger]"
                captions.append(caption)
                logger.debug("Final caption for image %d: %s", i + idx, caption)

            # Clear memory more aggressively
            for pil_image in batch_pil_images:
                pil_image.close()
            del inputs
            torch.cuda.empty_cache()

    finally:
        # Cleanup
        if 'model' in locals():
            model.to("cpu")
            del model
        if 'processor' in locals():
            del processor
        if device == "cuda":
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            logger.debug(f"Final GPU memory: {final_memory / 1024**2:.2f}MB")

    logger.info("Generated %d captions.", len(captions))
    return captions

async def create_dataset(images, captions):
    """Create temporary dataset from images and captions"""
    destination_folder = os.path.abspath(f"tmp_datasets/{uuid.uuid4()}")
    logger.info("Creating a dataset in folder: %s", destination_folder)
    os.makedirs(destination_folder, exist_ok=True)

    jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
    logger.debug("Creating metadata file at: %s", jsonl_file_path)
    
    # Use buffered writes for better I/O performance
    with open(jsonl_file_path, "w") as jsonl_file:
        # Process files in chunks
        chunk_size = 50
        for i in range(0, len(images), chunk_size):
            chunk_images = images[i:i + chunk_size]
            chunk_captions = captions[i:i + chunk_size]
            
            # Process chunks in parallel
            tasks = [resolve_image_path(img) for img in chunk_images]
            local_paths = await asyncio.gather(*tasks)
            
            for image_item, local_path in zip(chunk_images, local_paths):
                # Await the resolve_image_path call
                logger.debug("Copying %s to dataset folder %s", local_path, destination_folder)

                new_image_path = shutil.copy(local_path, destination_folder)
                file_name = os.path.basename(new_image_path)
                data = {"file_name": file_name, "prompt": chunk_captions[i - len(images) + i]}
                jsonl_file.write(json.dumps(data) + "\n")
                logger.debug("Wrote to metadata.jsonl: %s", data)

    # Verify dataset creation
    logger.debug("Dataset folder contents: %s", os.listdir(destination_folder))
    logger.debug("metadata.jsonl exists: %s", os.path.exists(jsonl_file_path))
    
    return destination_folder


async def resolve_image_path(image_item):
    """Handle multiple possible 'image' input formats"""
    if isinstance(image_item, dict) and "name" in image_item:
        return image_item["name"]

    if isinstance(image_item, str):
        # Check if it looks like an http(s) URL
        if image_item.lower().startswith(("http://", "https://")):
            os.makedirs("tmp_downloads", exist_ok=True)
            local_name = os.path.join("tmp_downloads", f"{uuid.uuid4()}.png")
            try:
                r = requests.get(image_item, stream=True)
                r.raise_for_status()
                with open(local_name, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return local_name
            except Exception as ex:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image from URL {image_item}: {ex}"
                )

        # Check if it's a local file path
        if os.path.exists(image_item):
            return image_item
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image path or URL: {image_item}"
            )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported image type or format: {image_item}"
    )

def recursive_update(d, u):
    """Recursively update nested dictionaries"""
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def check_local_model_files(model_type="dev"):
    """Check if model files exist locally and return the path"""
    base_path = os.path.join("models", "flux")
    model_path = os.path.join(base_path, "FLUX.1-schnell" if model_type == "schnell" else "FLUX.1")
    
    if not os.path.exists(model_path):
        logger.error(f"Model files not found at {model_path}")
        raise RuntimeError(f"Model files not found. Please ensure the model is downloaded to {model_path}")
    
    return model_path
def train_model(
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
    slugged_lora_name = slugify(lora_name)
    logger.info("Training LoRA model. Name: %s, Slug: %s", lora_name, slugged_lora_name)

    # Load default config
    logger.debug("Loading default config file: config/examples/train_lora_flux_24gb.yaml")
    with open("config/examples/train_lora_flux_24gb.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update configuration
    config["config"]["name"] = slugged_lora_name
    process_block = config["config"]["process"][0]
    process_block.update({
        "model": {
            "low_vram": low_vram,
            "name_or_path": "black-forest-labs/FLUX.1-schnell" if model_type == "schnell" else "black-forest-labs/FLUX.1",
            "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter" if model_type == "schnell" else None
        },
        "train": {
            "skip_first_sample": True,
            "steps": int(steps),
            "lr": float(lr)
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
        process_block["trigger_word"] = concept_sentence

    if sample_prompts:
        logger.debug("Sample prompts provided. Will enable sampling.")
        process_block["train"]["disable_sampling"] = False
        process_block["sample"].update({
            "sample_every": steps,
            "sample_steps": 28 if model_type == "dev" else 4,
            "prompts": sample_prompts
        })
    else:
        logger.debug("No sample prompts provided. Disabling sampling.")
        process_block["train"]["disable_sampling"] = True

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
    job = get_job(config_path)
    logger.debug("job object => %s", job)
    if job is None:
        raise RuntimeError(f"get_job() returned None for config path: {config_path}. Please check your job definition.")

    s3_folder_url = None # Initialize s3_folder_url outside try block

    try: # Added try-except block around job.run() and job.cleanup()
        job_start_time = time.time() # Timing start for job.run()
        logger.info("Running job...")
        job.run()
        job_runtime = time.time() - job_start_time # Calculate job runtime
        logger.info(f"Job run completed in {job_runtime:.2f} seconds.") # Log job runtime

        cleanup_start_time = time.time() # Timing start for job.cleanup()
        logger.info("Cleaning up job...")
        job.cleanup()
        cleanup_runtime = time.time() - cleanup_start_time # Calculate cleanup runtime
        logger.info(f"Job cleanup completed in {cleanup_runtime:.2f} seconds.") # Log cleanup runtime


        # Upload to S3
        bucket_name = os.environ.get("S3_BUCKET")
        s3_domain = os.getenv("S3_DOMAIN", "https://r2.syntx.ai")
        local_model_dir = f"output/{slugged_lora_name}"


        if bucket_name and os.path.exists(local_model_dir):
            s3_prefix = f"loras/flux/{slugged_lora_name}"
            logger.info("Uploading trained model to S3: bucket=%s, prefix=%s", bucket_name, s3_prefix)
            upload_start_time = time.time() # Timing start for S3 upload
            logger.info("Starting S3 upload...")
            if upload_directory_to_s3(local_model_dir, bucket_name, s3_prefix):
                upload_runtime = time.time() - upload_start_time # Calculate upload runtime
                logger.info(f"S3 upload completed in {upload_runtime:.2f} seconds.") # Log upload runtime

                # Construct an HTTP-based “folder” URL on Timeweb S3
                s3_endpoint = os.environ.get("S3_ENDPOINT", "https://s3.timeweb.cloud").rstrip("/")
                s3_folder_url = f"{s3_domain}/{s3_prefix}/"
                logger.info("Model folder successfully uploaded to: %s", s3_folder_url)
            else:
                raise RuntimeError("upload_directory_to_s3 returned False indicating failure.") # Explicitly raise error if upload fails
        else:
            logger.warning("No S3_BUCKET set or local_model_dir does not exist. Skipping upload.")
            raise RuntimeError("S3 bucket not configured or model directory missing, cannot complete training.") # Raise error as S3 URL is essential

    except Exception as e_train_job: # Catch exceptions from job.run(), job.cleanup(), or S3 upload
        logger.error(f"Error during training job execution in train_model: {e_train_job}", exc_info=True)
        raise RuntimeError(f"Training job failed: {e_train_job}") from e_train_job # Re-raise to be caught by train_lora's except

    finally: # Cleanup always, regardless of training success or failure, and *before* returning
        # Cleanup
        logger.debug("Removing dataset folder: %s", dataset_folder)
        shutil.rmtree(dataset_folder, ignore_errors=True)
        logger.debug("Removing config file: %s", config_path)
        os.remove(config_path)


    if not s3_folder_url: # Check again after the try-except-finally block, in case S3 upload failed inside try
        msg = "Failed to obtain S3 folder URL after training, possibly due to upload failure or S3 configuration issues."
        logger.error(msg)
        raise RuntimeError(msg)

    return s3_folder_url