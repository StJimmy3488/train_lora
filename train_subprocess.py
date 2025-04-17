import sys
import os
import json
import torch
import gc
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log everything except the result to stderr
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting training subprocess")
    try:
        # Read request from stdin
        logger.info("Reading request data from stdin")
        request_json = sys.stdin.readline()
        request_dict = json.loads(request_json)
        logger.debug(f"Received request: {request_dict}")

        # Setup GPU environment
        if torch.cuda.is_available():
            logger.info("CUDA is available, setting up GPU environment")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            logger.debug(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Import and run training
        logger.info("Importing training modules")
        from flux_train_api import train_lora
        import asyncio
        
        logger.info("Starting training process")
        result = asyncio.run(train_lora(**request_dict))
        logger.info(f"Training completed with result: {result}")

        # Output result to stdout
        print(json.dumps(result), flush=True)

    except Exception as e:
        import traceback
        error_info = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        logger.error(f"Training failed: {error_info}")
        # Make sure to output JSON to stdout even in case of error
        print(json.dumps(error_info), flush=True)
        sys.exit(1)

    finally:
        logger.info("Cleaning up resources")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main() 