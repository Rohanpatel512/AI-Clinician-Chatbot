import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os 
from pathlib import Path
import json 

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=api_key)

index_name = "medical-clinical-rag"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

index = pc.Index(index_name)

def _load_vector_db(chunks, batch_size=50):
    """
    Loads all chunks into vector database.
    Parameters:
        chunks (list) - List of chunks obtained through semantic chunking.
    Returns:
        None
    """

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]

        index.upsert_records(
            namespace="clinical_record",
            records=batch
        )


def search(user_query, k=10):
    """
    Performs semantic search on users query and gets the
    top 10 best answers and selects one with the highest
    score.
    
    Parameters:
      query (string) - Users query to be embedded during semantic search
      k (int) - Number of neighbors to "look at" when selecting best answer
    
    Returns:
      answer - Best answer given users query.
    """
    
    results = index.search(
        namespace="clinical_record",
        inputs = {
            "text": user_query 
        },
        top_k=k
    )

    result = [
        hit.fields["chunk_text"]
        for hit in results.result.hits
    ]
    
    return result[0]

if __name__ == "__main__":

    # Load all the chunks from JSON file
    data_path = REPO_ROOT / "data" / "rag_data.json"

    with open(data_path, "r") as file:
        chunks = json.load(file)
    
    _load_vector_db(chunks)