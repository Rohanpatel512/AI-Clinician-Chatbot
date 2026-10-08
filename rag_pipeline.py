# Imports 
from rag.vector_store import search 
from rag.augmentation import augment_prompt

def rag_pipeline(user_query, conversation_history):
    """
    Passes user query and conversation through RAG pipeline.
    Parameters:
      user_query (str) - Users current question.
      conversation_history (array) - An array of users conversation history with chatbot.
    
    Returns:
      response (str) - Model's final response to users query based on history, and RAG data.
    """

    # Retrieve the closest response to users query 
    retrieved_chunk = search(user_query)

    # Generate prompt to pass into LLM 
    prompt = augment_prompt(conversation_history, retrieved_chunk, user_query)

    #TODO: Pass prompt through trained model to get it's response

    #TODO: Return model response 