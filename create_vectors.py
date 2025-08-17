import os
import requests
from dotenv import load_dotenv

API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
headers = {
    "Authorization": f"Bearer {hf_token}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "Today is a sunny day and I will get some ice cream.",
})
def generate_embeddings(texts):
    embeddings = query({
        "inputs": texts,})
    return embeddings