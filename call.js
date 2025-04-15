import { Client } from "@gradio/client";
import fs from "fs";
import axios from "axios";
import yaml from "js-yaml";
import path from "path";
import os from "os";

// Load advanced options from an external YAML file
function loadYamlConfig(filePath) {
    try {
        const fileContents = fs.readFileSync(filePath, "utf8");
        return yaml.load(fileContents);
    } catch (error) {
        console.error("Error loading YAML file:", error);
        return null;
    }
}

// Path to the external YAML configuration file
const CONFIG_FILE = "advanced_options.yaml";

// Load configuration
const advancedOptions = loadYamlConfig(CONFIG_FILE);

// Define image URLs
const imageUrls = [
    "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=1024",
    "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=1024"
];

// Function to download an image from a URL and save it to a temporary directory
async function downloadImage(url, tempDir) {
    const response = await axios.get(url, { responseType: "arraybuffer" });
    const imageName = path.basename(url);
    const imagePath = path.join(tempDir, imageName);
    fs.writeFileSync(imagePath, response.data);
    return imagePath;
}

// Function to remove temporary files
function removeTempFiles(filePaths) {
    filePaths.forEach(filePath => {
        fs.unlinkSync(filePath);
    });
}

// Function to format the image data for Gradio
function formatImageData(imagePath) {
    return {
        data: fs.readFileSync(imagePath).toString("base64"), // Convert image to base64
        meta: { _type: "gradio.FileData" } // Add the required meta field
    };
}

async function main() {
    // Create a temporary directory
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "gradio-images-"));

    try {
        // Download images to the temporary directory
        const imagePaths = await Promise.all(imageUrls.map(url => downloadImage(url, tempDir)));

        // Format the image data for Gradio
        const formattedImages = imagePaths.map(imagePath => formatImageData(imagePath));

        // Connect to the Gradio client
        const client = await Client.connect("https://005cfef1293a5a9d7e.gradio.live/");

        // Make the prediction request with the formatted image data
        const result = await client.predict("/predict", { 
            images: formattedImages, 		
            lora_name: "Hello!!", 		
            concept_sentence: "Hello!!", 		
            steps: 1000, 		
            lr: 0.0004, 		
            rank: 16, 		
            model_type: "dev", 		
            low_vram: true, 		
            sample_prompts: null, 		
            advanced_options: advancedOptions, 
        });

        console.log(result.data);

    } catch (error) {
        console.error("Error during prediction:", error);
    } finally {
        // Remove temporary files after the request is done
        removeTempFiles(imagePaths);
        // Remove the temporary directory
        fs.rmdirSync(tempDir);
    }
}

// Run the main function
main();