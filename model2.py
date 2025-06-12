from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

def model_pipeline(image: Image.Image) -> str:
    # Load processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()

    # Preprocess input image
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


# saless force
# blip-image-captioning-base
# https://huggingface.co/Salesforce/blip-image-captioning-base