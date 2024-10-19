from fastapi import FastAPI
import gradio as gr

from app import interface

app = FastAPI()

@app.get('/')
async def root():
    return "gradio app running on /gradio"

app = gr.mount_gradio_app(app, interface, path='/gradio')