from typing import Union

from fastapi import FastAPI, UploadFile

from PIL import Image

import io

from model1 import model_pipeline
from model2 import model_pipeline



app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello, World!": "This is a fastapi AI app with huggingface dandelin vlit 32 b finetuned vqa"}


@app.post("/askvilt")
def ask(text: str, image: UploadFile):
    content = image.file.read()

    image = Image.open(io.BytesIO(content))
    # image  =Image(image.file)

    result = model_pipeline(text, image)

    return {"answer": result}


@app.post("/askblip")
def ask(image: UploadFile): 
    content = image.file.read()

    image = Image.open(io.BytesIO(content))
    # image  =Image(image.file)

    result = model_pipeline(image)

    return {"answer": result}
