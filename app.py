import gradio as gr
import PyPDF2
from create_index import upsert_embeddings, create_index
import uuid
from create_vectors import generate_embeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import cohere

load_dotenv()
PINECONE_KEY = os.getenv("PINECONE_KEY")
COHERE_KEY = os.getenv("COHERE_KEY")

pdf_text = ""
user_id = None
index = None

pc = Pinecone(
            api_key = PINECONE_KEY
        )
co = cohere.ClientV2(api_key=COHERE_KEY)


def extract_pdf_text(pdf_file):
    """
    This function extracts text from pdf files and returns the text.
    """
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def process_pdf(pdf_file):
    """
    This function extracts text from a pdf file and converts them into vector embeddings.
    These embeddings are then stored in a vector database for similarity search.
    """
    global pdf_text, user_id, index, pc
    if pdf_file is not None:
        pdf_text = extract_pdf_text(pdf_file)
    
    if pdf_text:
        user_id = str(uuid.uuid4())[:8]
        index = create_index(user_id, pc)
        upsert_embeddings(index = index, text=pdf_text)
    else:
        return "No file uploaded"

def generate_response(query, context):
    """
    This function takes a user query, converts it into vector embeddings using the same model used to convert the pdf document
    into embeddings, and then does a similarity search in the vector store.
    It uses the retrieved text from the document and generates a response using an llm model.
    """
    res = co.chat(
        model="command-r-plus-08-2024",
     messages=[
        {
          "role" : "system",
          "content" : f"Use only the following information : {context}, to answer user queries in a concise manner",
        },
        {
            "role": "user",
            "content": query,
        }
    ],
    )
    return res.message.content[0].text

def process_query(query):
    if index is None:
        return "Vector index not found"
    embeds = generate_embeddings([query])
    res = index.query(vector = embeds, top_k = 10, include_metadata=True)
    complete_text = ""
    for match in res['matches']:
        text = match['metadata']['text']
        complete_text += text + " "
    response = generate_response(query = query, context=complete_text)
    return response


def cleanup():
    """
    Function to delete the vector index after user clicks the cleanup button after all queries.
    """
    global user_id, pc, index
    if user_id is not None:
        index_name = f"user-index-{user_id}"
        pc.delete_index(name=index_name)
        result = "Cleanup completed"
        index = None
    else:
        result = "No user ID found for cleanup."
    return result

with gr.Blocks() as interface:
    with gr.Row():
        gr.Markdown("Upload your pdf document and ask away. Click cleanup after queries")
    with gr.Row():
        pdf_file_input = gr.File(label="Upload file")
        query_input = gr.Textbox(label="Enter your query")
    with gr.Row():
        submit_pdf_button = gr.Button("Submit PDF")
        submit_query_button = gr.Button("Submit query")
        cleanup_button = gr.Button("Cleanup")

    output = gr.Textbox(label="Output")

    # Set the button actions
    submit_pdf_button.click(fn=process_pdf, inputs=pdf_file_input, outputs=output)
    submit_query_button.click(fn=process_query, inputs=query_input, outputs=output)
    cleanup_button.click(fn=cleanup, outputs=output)

# Launch the interface
interface.launch(server_name="localhost", server_port=7860)