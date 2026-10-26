import os
import uvicorn
import gradio as gr
from backend.app.main import app as fastapi_app

# Satisfy Hugging Face ZeroGPU check if ZeroGPU hardware is selected
try:
    import spaces

    @spaces.GPU
    def _hf_spaces_gpu_ping():
        return True

    _hf_spaces_gpu_ping()
except Exception:
    pass

# Create a clean status dashboard for the Hugging Face Space web view
with gr.Blocks(title="HR Policy Bot API") as demo:
    gr.Markdown("## 🤖 HR Policy Bot — Agentic RAG API")
    gr.Markdown(
        "The backend API is **online** and serving REST endpoints for your Next.js frontend:\n\n"
        "- `GET /health` — Health check status\n"
        "- `GET /api/default-document` — Default HR policies metadata\n"
        "- `POST /api/query` — RAG question answering with reranking\n"
        "- `POST /api/upload` — In-memory PDF upload and indexing\n"
    )

# Mount Gradio interface onto FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

