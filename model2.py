from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Initialize global variables for model and processor
_processor = None
_model = None

def model_pipeline(image: Image.Image) -> str:
    global _processor, _model

    # Lazy load model and processor if not already loaded
    if _processor is None:
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    if _model is None:
        _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _model.eval()

    # Preprocess input image
    inputs = _processor(images=image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        out = _model.generate(**inputs)
        caption = _processor.decode(out[0], skip_special_tokens=True)

    return caption


# saless force
# blip-image-captioning-base
# https://huggingface.co/Salesforce/blip-image-captioning-base