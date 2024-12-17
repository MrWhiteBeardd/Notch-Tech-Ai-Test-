# Notch-Tech-Ai-Test-
This is the solution for the Test.
# YOLO + Image Captioning Pipeline

This repository demonstrates two image captioning pipelines that integrate **YOLOv8** for object detection and different models for generating captions: **BLIP** (for one pipeline) and **OpenAI GPT-3.5/4** (for the other). Both pipelines process images, detect objects, generate captions, and display results.

## Projects Overview

1. **YOLO + BLIP Image Captioning Pipeline**: This pipeline uses **YOLOv8** for detecting objects in images and **BLIP** (Bootstrapping Language-Image Pre-training) for generating captions based on the detected objects.
2. **YOLO + OpenAI GPT-3/4 Image Captioning Pipeline**: This pipeline integrates **YOLOv8** for object detection and OpenAI’s **GPT** model to generate more detailed and context-rich captions for the detected objects.

## Dependencies

You need to install the following libraries to run both pipelines:

```bash
pip install ultralytics transformers pillow matplotlib openai
```
## Setup OpenAI API Key (for GPT-based pipeline)

If you're using the OpenAI GPT-3/4 pipeline, you'll need to set up your OpenAI API key.

Obtain your API key from OpenAI.
In the yolo_openai_pipeline.py script, replace the placeholder API key with your actual OpenAI API key:

```bash
openai.api_key = "your_openai_api_key"
```
the key used is 
```bash
openai.api_key = os.getenv("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")

```

## Code Walkthrough
YOLO + BLIP Image Captioning Pipeline (yolo_blip_pipeline.py)
Features
YOLOv8: Detects objects in the images.
BLIP: Generates captions based on the content of the images.
Combining Results: Merges detected objects with captions generated by the BLIP model.
Image Display: Displays images alongside captions using matplotlib.
How It Works
Object Detection: YOLOv8 detects objects in the image.
Caption Generation: The BLIP model generates captions based on the image content.
Results Display: Displays the images and captions together.
Example Usage
Prepare Your Images: Place the image files (e.g., 1.jpg, 2.jpg) in the same directory as the script.
# Output
The script will:
Print detected objects to the terminal.
Show images along with the captions using matplotlib.
## YOLO + OpenAI GPT-3/4 Image Captioning Pipeline (yolo_openai_pipeline.py)
Features
YOLOv8: Detects objects in images.
OpenAI GPT-3/4: Generates detailed captions for the detected objects.
Keyword Integration: Optionally include a user-defined keyword to enrich captions.
Image Display: Displays images alongside captions using matplotlib.
How It Works
Object Detection: YOLOv8 detects objects in the image.
Caption Generation: GPT generates a detailed caption based on detected objects, optionally including a user-defined keyword.
Results Display: Displays the images with captions.
# Output
The script will:

Print detected objects to the terminal.
Display images with captions in a matplotlib window.
