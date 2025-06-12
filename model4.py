from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from typing import Union, List

# Initialize global variables for model and processors
_model = None
_feature_extractor = None
_tokenizer = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_length = 16
# Remove beam search parameters and use greedy search instead
gen_kwargs = {"max_length": max_length}

def model_pipeline(image: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> str:
    global _model, _feature_extractor, _tokenizer

    # Lazy load model and processors if not already loaded
    if _model is None:
        _model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        _model.to(device)
    if _feature_extractor is None:
        _feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    """
    Generate image caption using ViT-GPT2 model.
    
    Args:
        image: Can be a single image path (str), PIL Image object, or a list of either
        
    Returns:
        str: Generated caption for the image
    """
    # Convert single input to list for consistent processing
    if not isinstance(image, list):
        image = [image]
    
    # Process images
    images = []
    for img in image:
        if isinstance(img, str):
            # If input is a path, open the image
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        images.append(img)
    
    # Generate captions
    pixel_values = _feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    output_ids = _model.generate(pixel_values, **gen_kwargs)
    preds = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    # Return single caption if single image was provided
    return preds[0] if len(preds) == 1 else preds


# vit-gpt2 image captioning -- nlpconnect
# https://huggingface.co/nlpconnect/vit-gpt2-image-captioning


