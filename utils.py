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

# Add at the top of utils.py
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ["NCCL_P2P_DISABLE"] = "0"
os.environ["NCCL_IB_DISABLE"] = "0"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

def process_images_and_captions(images, concept_sentence=None):
    """Process uploaded images and generate captions if needed"""
    logger.debug("Processing images for captioning. Number of images: %d", len(images))

    # if len(images) < 2:
    #     raise gr.Error("Please upload at least 2 images")
    # elif len(images) > 150:
    #     raise gr.Error("Maximum 150 images allowed")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16

    logger.info("Loading AutoModelForCausalLM from 'multimodalart/Florence-2-large-no-flash-attn'.")

    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn",
        trust_remote_code=True
    )

    logger.debug("processor.post_process_generation => %s", processor.post_process_generation)
    captions = []
    try:
        for image_path in images:
            logger.debug("Processing image: %s", image_path)
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
                image.load()

            prompt = "<DETAILED_CAPTION>"
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(device, torch_dtype)

            logger.debug("Generating tokens with model.generate(...)")
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
            logger.debug("Generated text: %s", generated_text)

            try:
                logger.debug("Calling processor.post_process_generation(...)")
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task=prompt,
                    image_size=(image.width, image.height)
                )
            except Exception as ex:
                logger.error("Error calling post_process_generation: %s", ex, exc_info=True)
                raise

            logger.debug("Parsed answer: %s", parsed_answer)

            caption = parsed_answer["<DETAILED_CAPTION>"].replace(
                "The image shows ", ""
            )
            if concept_sentence:
                caption = f"{caption} [trigger]"
            captions.append(caption)
            logger.debug("Final caption for image: %s", caption)
    finally:
        logger.debug("Cleaning up model and processor from GPU/Memory.")
        model.to("cpu")
        del model
        del processor

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

async def recursive_update(d, u):
    """Recursively update nested dictionaries"""
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = await recursive_update(d.get(k, {}), v)
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
    config_path = None
    s3_folder_url = None
    local_model_dir = None

    try:
        slugged_lora_name = slugify(lora_name)
        logger.info("Training LoRA model. Name: %s, Slug: %s", lora_name, slugged_lora_name)

        # Load default config
        logger.debug("Loading default config file: config/examples/train_lora_flux_24gb.yaml")
        with open("config/examples/train_lora_flux_24gb.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Get absolute path for dataset folder
        dataset_folder = os.path.abspath(dataset_folder)
        
        # Update configuration
        config["config"]["name"] = slugged_lora_name
        process_block = config["config"]["process"][0]
        
        # Configure dataset with absolute path and metadata file
        process_block["datasets"] = [{
            "folder_path": dataset_folder,
            "metadata_file": "metadata.jsonl",
            "cache_to_disk": True,
            "load_in_memory": True,
            "shuffle": True,
            "num_workers": 2,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "pin_memory": True,
            "multiprocessing_context": "spawn",
            "drop_last": True
        }]

        # Configure for single-process operation
        process_block["train"].update({
            "dataloader_workers": 2,
            "dataloader_timeout": 120,
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "bf16",
            "seed": 42,
            "pin_memory": True,
            "gradient_checkpointing": True,
            "use_cached_latents": True,
            "prefetch_factor": 2,
            "use_deterministic_algorithms": False
        })

        # Optimize for high-end GPU
        process_block["environment"] = {
            "multiprocessing_context": "spawn",
            "torch_compile": False,
            "torch_inference_mode": False,
            "cudnn_benchmark": True,
            "deterministic_algorithms": False,
            "cuda_launch_blocking": "0",
            "cuda_device": "0",
            "cuda_memory_fraction": 0.95,
            "worker_init": True,
            "shared_memory": True
        }

        # Log the dataset configuration for debugging
        logger.debug("Dataset configuration: %s", process_block["datasets"][0])
        logger.debug("Dataset folder exists: %s", os.path.exists(dataset_folder))
        logger.debug("Dataset folder contents: %s", os.listdir(dataset_folder))
        
        process_block.update({
            "model": {
                "low_vram": False,
                "attention_implementation": "flash_attention_2",
                "gradient_checkpointing": False,
                "enable_xformers_memory_efficient_attention": True
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
            "save": {
                "output_dir": f"tmp_models/{slugged_lora_name}",
                "push_to_hub": False
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
                "prompts": sample_prompts.split(",") if isinstance(sample_prompts, str) else sample_prompts
            })
        else:
            logger.debug("No sample prompts provided. Disabling sampling.")
            process_block["train"]["disable_sampling"] = True

        if advanced_options:
            logger.debug("Merging advanced_options YAML into config.")
            if isinstance(advanced_options, str):
                advanced_options_dict = yaml.safe_load(advanced_options)
                # Preserve critical settings
                train_config = process_block["train"].copy()
                env_config = process_block.get("environment", {}).copy()
                config["config"]["process"][0] = await recursive_update(
                    config["config"]["process"][0],
                    advanced_options_dict
                )
                # Restore critical settings
                process_block["train"].update(train_config)
                process_block["environment"] = env_config

        # Save config
        os.makedirs("tmp_configs", exist_ok=True)
        config_path = f"tmp_configs/{uuid.uuid4()}-{slugged_lora_name}.yaml"
        logger.debug("Saving updated config to: %s", config_path)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Run training
        logger.info("Retrieving job with config path: %s", config_path)
        job = get_job(config_path, slugged_lora_name)
        logger.debug("job object => %s", job)

        if job is None:
            raise RuntimeError(f"get_job() returned None for config path: {config_path}")


        logger.info("Running job...")
        job.run()
        logger.info("Job completed, cleaning up...")
        job.cleanup()

        # Upload to S3
        local_model_dir = f"output/{slugged_lora_name}"
        bucket_name = os.environ.get("S3_BUCKET")
        s3_domain = os.getenv("S3_DOMAIN", "https://r2.syntx.ai")

        if bucket_name and os.path.exists(local_model_dir):
            s3_prefix = f"loras/flux/{slugged_lora_name}"
            logger.info("Uploading trained model to S3: bucket=%s, prefix=%s", bucket_name, s3_prefix)
            
            if await upload_directory_to_s3(local_model_dir, bucket_name, s3_prefix):
                s3_folder_url = f"{s3_domain}/{s3_prefix}/"
                logger.info("Model folder successfully uploaded to: %s", s3_folder_url)
            else:
                raise RuntimeError("Failed to upload model to S3")
        else:
            raise RuntimeError("S3 bucket not configured or model directory missing")

        return s3_folder_url

    except Exception as e:
        logger.error("Error in train_model: %s", str(e))
        raise

    finally:
        # Always clean up temporary files
        try:
            # Clean up dataset folder
            if dataset_folder and os.path.exists(dataset_folder):
                logger.debug("Removing dataset folder: %s", dataset_folder)
                shutil.rmtree(dataset_folder, ignore_errors=True)
            
            # Clean up config file
            if config_path and os.path.exists(config_path):
                logger.debug("Removing config file: %s", config_path)
                os.remove(config_path)
            
            # Clean up local model directory if upload failed
            if local_model_dir and os.path.exists(local_model_dir) and not s3_folder_url:
                logger.debug("Removing local model directory: %s", local_model_dir)
                shutil.rmtree(local_model_dir, ignore_errors=True)
            
            # Clean up any tmp_downloads directory
            if os.path.exists("tmp_downloads"):
                logger.debug("Removing tmp_downloads directory")
                shutil.rmtree("tmp_downloads", ignore_errors=True)
                
        except Exception as cleanup_error:
            logger.error("Error during cleanup: %s", str(cleanup_error))

def monitor_gpu():
    if torch.cuda.is_available():
        logger.info(f"GPU utilization: {torch.cuda.utilization()}%")
        logger.info(f"Memory used: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
        logger.info(f"Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")

def monitor_training_performance():
    if torch.cuda.is_available():
        logger.info(f"GPU utilization: {torch.cuda.utilization()}%")
        logger.info(f"Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        logger.info(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        logger.info(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")

def create_dataloader(dataset, config):
    """Create a properly configured DataLoader"""
    try:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            persistent_workers=config["persistent_workers"],
            prefetch_factor=config["prefetch_factor"],
            pin_memory=config["pin_memory"],
            shuffle=config["shuffle"],
            multiprocessing_context='spawn',
            drop_last=True
        )
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}")
        # Fallback to safe settings
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=0,
            shuffle=config["shuffle"]
        )