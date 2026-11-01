from fastapi import APIRouter, Depends, HTTPException, Request
import json 
from app.services.rag_pipeline import rag_pipeline, contextualize_query, create_chat_history

chat_router = APIRouter(prefix="/chat", tags=["Chat"])

@chat_router.post("/send_message")
async def send_message(request: Request):

    data = await request.body()

    # Convert the bytes object data into a JSON
    data = json.loads(data)

    user_query = data['message']
    conversation_history = data['conversation_history']
    first_turn = data['first_turn']

    if user_query is None or user_query.strip() == "":
        return {"error": "User message is empty."}


    # Rewrite user query based on conversation history unless the query is standalone
    response = contextualize_query(user_query, conversation_history)

    chat_history = create_chat_history(conversation_history)

    # Send the message through the rag pipeline to recieve a response
    model_response = await rag_pipeline(user_query, response.rewritten_query, response.urgency, conversation_history, first_turn)
    
    if isinstance(model_response, dict) and "response" in model_response:
        return {"response": model_response["response"]}
    if isinstance(model_response, str):
        return {"response": model_response}

    # Returns LLM response
    return {"response": str(model_response)}
