import sys
import os
import json
import torch
import gc
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

# Add paths
sys.path.extend([os.getcwd(), "ai-toolkit"])

def main():
    try:
        # Read request from stdin
        request_json = sys.stdin.read()
        request_dict = json.loads(request_json)

        # Setup GPU environment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()

        # Import and run training
        from flux_train_api import train_lora
        import asyncio
        result = asyncio.run(train_lora(**request_dict))

        # Output result
        print(json.dumps(result))

    except Exception as e:
        import traceback
        print(json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), file=sys.stderr)
        sys.exit(1)

    finally:
        # Cleanup GPU memory before exit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main() 