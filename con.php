<?php

function downloadImage($url) {
    $tempFile = tempnam(sys_get_temp_dir(), 'img') . '.jpg';
    $ch = curl_init($url);
    
    $fp = fopen($tempFile, 'wb');
    curl_setopt($ch, CURLOPT_FILE, $fp);
    curl_setopt($ch, CURLOPT_HEADER, 0);
    curl_exec($ch);
    
    if (curl_errno($ch)) {
        throw new Exception("Failed to download image: " . curl_error($ch));
    }
    
    curl_close($ch);
    fclose($fp);
    
    return $tempFile;
}

// Embedded YAML Configuration (Converted to JSON)
$advancedOptions = json_encode([
    "device" => "cuda:0",
    "model" => [
        "is_flux" => true,
        "quantize" => true,
        "name_or_path" => "/app/ai-toolkit/FLUX.1-dev"
    ],
    "network" => [
        "linear" => 16,
        "linear_alpha" => 16,
        "type" => "lora",
        "network_kwargs" => [
            "only_if_contains" => [
                "transformer.single_transformer_blocks.10.",
                "transformer.single_transformer_blocks.25."
            ]
        ]
    ],
    "sample" => [
        "sampler" => "flowmatch",
        "sample_every" => 1000,
        "width" => 1024,
        "height" => 1024,
        "prompts" => ["person in bustling cafe"]
    ],
    "save" => [
        "dtype" => "float16",
        "hf_private" => true,
        "max_step_saves_to_keep" => 4,
        "save_every" => 10000
    ],
    "train" => [
        "steps" => 10,
        "batch_size" => 1,
        "dtype" => "bf16",
        "ema_config" => [
            "ema_decay" => 0.99,
            "use_ema" => true
        ],
        "gradient_accumulation_steps" => 1,
        "gradient_checkpointing" => true,
        "lr" => 1e-3,
        "skip_first_sample" => true,
        "noise_scheduler" => "flowmatch",
        "optimizer" => "adamw8bit",
        "train_text_encoder" => false,
        "train_unet" => true
    ]
]);

// Image URLs
$imageUrls = [
    "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=1024",
    "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=1024"
];

// Download images
$imagePaths = array_map('downloadImage', $imageUrls);

// Prepare file uploads
$fileUploads = [];
foreach ($imagePaths as $index => $path) {
    $fileUploads["file$index"] = new CURLFile($path, "image/jpeg", basename($path));
}

// API request URL
$apiUrl = "https://005cfef1293a5a9d7e.gradio.live/gradio_api/call/predict";

// Correctly structure the multipart request
$postFields = array_merge($fileUploads, [
    "data" => $advancedOptions // JSON-encoded data
]);

// Initialize cURL request
$ch = curl_init($apiUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, $postFields);
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    "Content-Type: multipart/form-data"
]);

// Execute request
$response = curl_exec($ch);

// Check for cURL errors
if (curl_errno($ch)) {
    throw new Exception("API Request Failed: " . curl_error($ch));
}

// Close cURL session
curl_close($ch);

// Print API response
echo "API Response: " . $response . PHP_EOL;

// Clean up temporary image files
foreach ($imagePaths as $path) {
    unlink($path);
}

?>
