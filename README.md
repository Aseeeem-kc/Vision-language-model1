# Vision-Language Models FastAPI Application

This repository hosts a FastAPI application demonstrating the capabilities of various vision-language models for tasks such as Visual Question Answering (VQA) and Image Captioning.

## ✨ Features & Endpoints

-   **`/askvilt`**: Visual Question Answering (VQA) using a fine-tuned VLiT (Vision-and-Language Transformer) model. Expects a text query and an image.
-   **`/askblip`**: Image Captioning using a BLIP (Bootstrapping Language-Image Pre-training) model. Expects an image.
-   **`/askclip`**: Image Embedding/Search using a CLIP (Contrastive Language-Image Pre-training) model. Expects an image.
-   **`/askvitgpt2`**: Image Captioning using a ViT-GPT2 (Vision Transformer and GPT-2) model. Expects an image.

## 🚀 Getting Started

Follow these steps to set up and run the application locally:

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/vision-language-model.git
    cd vision-language-model
    ```
    (Replace `your-username/vision-language-model.git` with your actual repository URL)

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    .\env\Scripts\activate   # On Windows
    # source env/bin/activate  # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Running the Application

Once the dependencies are installed, you can start the FastAPI server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reloading on code changes, `--host 0.0.0.0` makes the server accessible externally, and `--port 8000` sets the listening port.

Your API will be accessible at `http://localhost:8000`.

## 💡 Usage

To interact with the API and test the endpoints, navigate to the interactive documentation (Swagger UI) provided by FastAPI in your browser:

[http://localhost:8000/docs](http://localhost:8000/docs)

From there, you can explore each endpoint, view expected parameters, and make test requests directly from your browser.

## 📄 License

This project is open source and available under the [MIT License](LICENSE). 
