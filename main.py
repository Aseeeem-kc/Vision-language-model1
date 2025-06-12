from typing import Union

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import io

from model1 import model_pipeline as vlit_model
from model2 import model_pipeline as blip_model
from model3 import model_pipeline as clip_model
from model4 import model_pipeline as vit_gpt2_model



app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def read_root():
    return {"Hello, World!": "This is a fastapi AI app with huggingface dandelin vlit 32 b finetuned vqa"}


@app.post("/askvilt")
def ask(text: str, image: UploadFile):
    content = image.file.read()

    image = Image.open(io.BytesIO(content))
    # image  =Image(image.file)

    result = vlit_model(text, image)

    return {"answer": result}


@app.post("/askblip")
def ask(image: UploadFile): 
    content = image.file.read()

    image = Image.open(io.BytesIO(content))
    # image  =Image(image.file)

    result = blip_model(image)

    return {"answer": result}


@app.post("/askclip")
def ask(image: UploadFile):
    content = image.file.read()

    image = Image.open(io.BytesIO(content))

    result = clip_model(image)

    return {"asnwer":result}


@app.post("/askvitgpt2")
def ask(image: UploadFile):
    content = image.file.read()

    image = Image.open(io.BytesIO(content))

    result = vit_gpt2_model(image)

    return {"answer": result}