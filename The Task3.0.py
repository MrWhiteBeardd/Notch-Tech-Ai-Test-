import os
from ultralytics import YOLO
import openai
from PIL import Image
import matplotlib.pyplot as plt

# Set OpenAI API Key
openai.api_key = "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"

# Configuration
YOLO_MODEL_PATH = "yolov8n.pt"  # Path to YOLOv8 model weights
IMAGE_PATHS = ["1.jpg", "2.jpg", "3.jpg"]  # Replace with actual image file paths


# Functions
def load_yolo_model():
    """Load the pre-trained YOLOv8 model."""
    return YOLO(YOLO_MODEL_PATH)


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


def generate_caption(objects_detected, user_keyword=None):
    """Generate a caption using OpenAI GPT-3.5 or GPT-4 API."""
    try:
        base_prompt = f"The image contains the following objects: {', '.join(objects_detected)}."
        if user_keyword:
            base_prompt += f" Include the keyword '{user_keyword}' in the caption."

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-3.5-turbo" or "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are an assistant that generates image captions."},
                {"role": "user", "content": base_prompt + " Provide a detailed, context-rich caption for the image."}
            ],
            max_tokens=100
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating caption: {str(e)}"


def display_images_with_captions(image_paths, captions):
    """Display images with captions."""
    plt.figure(figsize=(12, 4 * len(image_paths)))
    for i, (image_path, caption) in enumerate(zip(image_paths, captions)):
        img = Image.open(image_path)
        plt.subplot(len(image_paths), 1, i + 1)
        plt.imshow(img)
        plt.title(caption, fontsize=10, wrap=True)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def image_caption_pipeline(image_paths, keyword=None):
    """Run the pipeline to detect objects and generate captions."""
    model = load_yolo_model()
    captions = []

    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        detected_objects = detect_objects(model, image_path)
        print(f"Detected Objects: {detected_objects}")
        caption = generate_caption(detected_objects, keyword)
        print(f"Caption: {caption}\n")
        captions.append(caption)

    display_images_with_captions(image_paths, captions)


# Main Execution
if __name__ == "__main__":
    # Run the pipeline
    image_caption_pipeline(IMAGE_PATHS, keyword="summer")
