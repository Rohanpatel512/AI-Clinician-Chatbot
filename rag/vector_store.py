import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os 
from pathlib import Path
import json 
import torch 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"

load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

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

def _load_vector_db(chunks: list, batch_size=50) -> None:
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

def rerank(query, documents, k=3) -> list:
    """
    Reranks the top k selected documents based of context-level matching

    Parameters:
        query - User's latest query (str)
        documents - All top-k retrieved chunks from vector DB (list)
        k - Top k number of documents to return (int)
    
    Returns:
        ranked - Returns top-k matching documents (list)
        logits - The scores for each passage/document (list)
    """

    # Pair up each chunk of text retrieved with query 
    pairs = [[query, document] for document in documents]

    with torch.no_grad():
        encoded = tokenizer(
            pairs,
            truncation=True, 
            padding=True,
            return_tensors="pt",
            max_length=512
        )

        logits = model(**encoded).logits.squeeze(dim=1)

    logits = logits.tolist()
    ranked = sorted(
        zip(documents, logits),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:k], logits

def search(user_query: str, k=10, threshold=9.0) -> str:
    """
    Performs semantic search on users query and gets the
    top 10 best answers and selects one with the highest
    score.
    
    Parameters:
      query (string) - Users query to be embedded during semantic search
      k (int) - Number of neighbors to "look at" when selecting best answer
      threshold (float) - Value score of highest ranked passage is compared against 
      
    Returns:
      answer (str) - Best answer given users query.
    """
    
    results = index.search(
        namespace="clinical_record",
        inputs = {
            "text": user_query 
        },
        top_k=k
    )

    # Gather all top-k documents into an array 
    documents = [hit.fields["chunk_text"] for hit in results.result.hits]

    # Rerank the documents 
    ranked, logits = rerank(user_query, documents)

    if not ranked:
        return ""

    best_passage, best_score = ranked[0]

    if best_score > threshold:
        return best_passage

    return ""

if __name__ == "__main__":    

    
    # Load all the chunks from JSON file
    data_path = REPO_ROOT / "data" / "rag_data.json"

    with open(data_path, "r") as file:
        chunks = json.load(file)
    
    _load_vector_db(chunks)
    