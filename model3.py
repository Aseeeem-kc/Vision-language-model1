"""
Reference implementation and examples:

# Original hardcoded image URL and labels
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# Original comments for reference
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
"""

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Initialize model and processor globally to avoid reloading
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def model_pipeline(image: Image.Image) -> str:
    """
    Process an image using CLIP model and return the most likely label.
    
    Args:
        image (Image.Image): Input PIL image
        
    Returns:
        str: Predicted label for the image
    """
    # Common image labels to check against
    labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a person",
        "a photo of a car",
        "a photo of a building",
        "a photo of nature",
        "a photo of food"
    ]
    
    # Process the image and labels
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    
    # Get model predictions
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # Get the index of highest probability
    best_idx = probs.argmax().item()
    
    # Return the corresponding label
    return labels[best_idx]

# Model reference: openai/clip-vit-base-patch32
# https://huggingface.co/openai/clip-vit-base-patch32

