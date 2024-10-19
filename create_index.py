from pinecone import ServerlessSpec
from create_vectors import generate_embeddings

def segment_text(text):
    segments = text.split('.')
    return segments

def create_index(user_id, pc):
    """
    function to instantiate a vector index in the vector store database in the cloud server.
    """
    index_name = f"user-index-{user_id}"
    if index_name not in pc.list_indexes():
        pc.create_index(name=index_name, dimension = 384, metric='cosine', spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) )
    return pc.Index(index_name)

def upsert_embeddings(index, text):
    """
    Function to segment text at sentence level and create embeddings to store in the vector store.
    """
    segments = segment_text(text) 
    batch_size = 5
    
    for i in range(0, len(segments), batch_size):
        i_end = min(i + batch_size, len(segments))
        texts = segments[i : i_end]
        
        try:
            embeds = generate_embeddings(texts) 
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            continue

        meta = [{'text': text} for text in texts] 
        to_upsert = [(f'id_{i}_{j}', embed, meta[j]) for j, embed in enumerate(embeds)]
        
        index.upsert(vectors=list(to_upsert))