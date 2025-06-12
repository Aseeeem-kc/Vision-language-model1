from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# Initialize global variables for model and processor
_processor = None
_model = None

# # prepare image + question
# url = "https://media.istockphoto.com/id/147290529/photo/emperors.jpg?s=612x612&w=0&k=20&c=ZApZFJtKoXGKYYJsgNcNPTMHqqSbbAx9CBg2AF2qyJk="
# image = Image.open(requests.get(url, stream=True).raw)
# text = "How many penguins are there?"

def model_pipeline(text: str, image: Image):
    global _processor, _model

    # Lazy load model and processor if not already loaded
    if _processor is None:
        _processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    if _model is None:
        _model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = _processor(image, text, return_tensors="pt")

    # forward pass
    outputs = _model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return _model.config.id2label[idx]


# vilt-b32-finetuned-vqa
# by dandelin
# https://huggingface.co/dandelin/vilt-b32-finetuned-vqa

