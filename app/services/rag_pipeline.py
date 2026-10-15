# Imports 
from rag.vector_store import search 
from rag.augmentation import augment_prompt
from fastapi import HTTPException
import httpx 
import os 
from app.core.config import MODAL_API_URL
import asyncio

async def rag_pipeline(user_query: str, conversation_history: list) -> str:
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

    #Pass prompt through trained model to get it's response
    endpoint = f"{MODAL_API_URL}" 

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                endpoint, 
                json={"prompt": prompt},
                headers={
                    "Modal-Key": os.environ["MODAL_KEY"],
                    "Modal-Secret": os.environ["MODAL_SECRET"]
                },
                timeout=400,
            )
            response.raise_for_status()
            response_data = response.json()
            if isinstance(response_data, dict) and "response" in response_data:
                return {"response": response_data["response"]}
            if isinstance(response_data, str):
                return {"response": response_data}
            return {"response": str(response_data)}
        
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Modal Error: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Failed to reach Modal: {str(e)}")