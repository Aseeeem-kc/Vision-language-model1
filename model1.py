from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# # prepare image + question
# url = "https://media.istockphoto.com/id/147290529/photo/emperors.jpg?s=612x612&w=0&k=20&c=ZApZFJtKoXGKYYJsgNcNPTMHqqSbbAx9CBg2AF2qyJk="
# image = Image.open(requests.get(url, stream=True).raw)
# text = "How many penguins are there?"

def model_pipeline(text: str, image: Image):

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]


# vilt-b32-finetuned-vqa
# by dandelin

