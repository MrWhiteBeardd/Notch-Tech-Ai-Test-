import os
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt

# Configuration
YOLO_MODEL_PATH = "yolov8n.pt"  # Path to YOLOv8 model weights
IMAGE_PATHS = ["1.jpg", "2.jpg", "3.jpg"]  # Replace with your actual image file paths

# Load YOLO Model
def load_yolo_model():
    """Load the pre-trained YOLOv8 model."""
    return YOLO(YOLO_MODEL_PATH)

# Object Detection
def detect_objects(model, image_path):
    """Perform object detection and return detected object names."""
    results = model(image_path)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            detected_objects.append(cls_name)
    return detected_objects

# Load Hugging Face BLIP Model for Captioning
def load_captioning_model():
    """Load the BLIP image captioning model."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Generate Caption Using BLIP
def generate_caption(image_path, processor, model):
    """Generate a caption for the image using BLIP model."""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

# Display Images with Captions
def display_images_with_captions(image_paths, captions):
    """Display images with generated captions."""
    plt.figure(figsize=(12, 4 * len(image_paths)))
    for i, (image_path, caption) in enumerate(zip(image_paths, captions)):
        img = Image.open(image_path)
        plt.subplot(len(image_paths), 1, i + 1)
        plt.imshow(img)
        plt.title(caption, fontsize=10, wrap=True)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main Pipeline
def image_caption_pipeline(image_paths):
    """Run the full pipeline: object detection and image captioning."""
    # Load models
    yolo_model = load_yolo_model()
    processor, caption_model = load_captioning_model()

    captions = []

    # Process images
    for image_path in image_paths:
        print(f"Processing image: {image_path}")

        # Object detection
        detected_objects = detect_objects(yolo_model, image_path)
        print(f"Detected Objects: {detected_objects}")

        # Generate image caption
        caption = generate_caption(image_path, processor, caption_model)
        if detected_objects:
            caption = f"The image contains {', '.join(detected_objects)}. {caption}"
        print(f"Caption: {caption}\n")

        captions.append(caption)

    # Display results
    display_images_with_captions(image_paths, captions)

# Main Execution
if __name__ == "__main__":
    # Run the pipeline
    image_caption_pipeline(IMAGE_PATHS)
