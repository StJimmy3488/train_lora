import requests
import yaml
from gradio_client import Client, handle_file
import tempfile
import os

# Load advanced options from an external YAML filex
def load_yaml_config(file_path):
    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)

# Path to the external YAML configuration file
CONFIG_FILE = "advanced_options.yaml"

# Load configuration
advanced_options = load_yaml_config(CONFIG_FILE)

# Initialize the client
client = Client("https://acfca9fcb21bbe293a.gradio.live")

# Define image URLs
image_urls = [
    "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=1024",
    "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=1024",
    "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=1024",
    "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=1024",
    "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=1024",
    "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=1024"
]

# Download images and save them as temporary files
def download_image(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        with open(temp_file.name, "wb") as file:
            file.write(response.content)
        return temp_file.name  # Return the file path
    else:
        raise Exception(f"Failed to download image from {url}")

# Process image URLs into local temporary files
image_paths = [download_image(url) for url in image_urls]  # Get file paths
image_files = [handle_file(path) for path in image_paths]  # Convert to Gradio format

# Convert YAML data to string format
def yaml_to_string(yaml_data):
    return yaml.dump(yaml_data, default_flow_style=False)

# Send the API request using properly formatted images
result = client.predict(
    images=image_files,  # Pass Gradio-wrapped file references
    lora_name="Hello!!",
    concept_sentence=None,
    steps=1000,
    lr=0.0004,
    rank=16,
    model_type="dev",
    low_vram=False,
    sample_prompts=None,
    advanced_options=yaml_to_string(advanced_options),
    api_name="/predict"
)

# Print the result
print(result)

# Clean up temporary files
for img_path in image_paths:  # Use the original downloaded file paths
    os.remove(img_path)

if __name__ == "__main__":
    pass
