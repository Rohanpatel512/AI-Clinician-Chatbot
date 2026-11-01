# Imports 
from rag.vector_store import search 
from rag.augmentation import augment_prompt
from app.core.schemas import QueryProcessor
from fastapi import HTTPException
import httpx 
import os 
from app.core.config import MODAL_API_URL, REWRITER_REPO_ID, REWRITER_FILE_NAME, OPEN_AI_KEY
import torch
import asyncio
import openai
import traceback

client = openai.OpenAI(api_key=OPEN_AI_KEY)

def contextualize_query(raw_query: str, conversation_history: list) -> str:
    """
    Uses a fast LLM to rewrite ambiguous user query into a standalone query
    based on the conversational history.

    Parameters:
        raw_query (str) - Users latest query.
        conversation_history (list) - An array of users conversation history with chatbot
    
    Returns:
        response (str) - A standalone query based on conversational history
    """

    # Create the chat history 
    chat_history = create_chat_history(conversation_history)

    # Formulate the prompt
    
    prompt = f"""
    You are a dual-purpose medical AI assistant. Your tasks are to rewrite a patient's message into a standalone search query AND classify their clinical urgency for safety routing.

    TASK 1: REWRITING RULES
    - You rewrite patient messages into standard medical search queries in the same point of view as user using the recent conversation history.
    - Detect contextual follow-ups, not just pronouns. Messages like "What about at night?", "And for kids?", or "Is that serious?" may depend on prior context.    
    - If the question is already standalone, do not rewrite it, keep it as it is.
    - Use relevant conversation history references, including symptoms, conditions, medications, treatments, and topics mentioned in prior answers.
    - Preserve the patient's intent and distinguish follow-ups, added details, and new topics. Include only context relevant to the current question.
    - Never invent medical facts, assume a diagnosis, or treat assistant-mentioned possibilities as confirmed facts.
    - If intended reference is genuinely unclear, do not guess; preserve the ambiguity so clarification can be requested.
    - Do not answer the medical question. Generate only the concise, standalone search query, with no explanation, inside the 'rewritten_query' field.

    TASK 2: INTENT AND URGENCY CLASSIFICATION
    - INTENT: Classify as ACTIVE_CONCERN if user describes active/recent symptoms happening to a person. Classify as GENERAL if it is general knowledge, definitions, facts, or hypothetical scenarios with no active patient. 
    - URGENCY: Classify as CRITICAL for life-threatening emergencies. Classify as ACUTE if they need prompt assessment within hours but are stable. Classify as SUBACUTE for mild/chronic symptoms managed with routine care. Use NONE if intent is GENERAL.

    Provide maximum objective focus on 'Latest Patient Message' to catch sudden clinical escalations.
    """

    content = f"""
    Conversation History:\n
    {chat_history}

    Latest Patient Message:\n
    {raw_query}
    """

    try:
        response = client.beta.chat.completions.parse(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content}
            ],
            response_format=QueryProcessor,
            temperature=0.0
        )

        result = response.choices[0].message

        if result.refusal:
            raise ValueError(f"Model refused: {result.refusal}")
        if result.parsed is None:
            raise ValueError("Model return no parsed output.")

    except openai.APIConnectionError as e:
        print("Connection failed:", repr(e.__cause__))
        print("Underlying cause:")
        traceback.print_exception(e.__cause__)

    except openai.APIStatusError as e:
        print("API status:", e.status_code)
        print("API error:", e.message)


    return result.parsed


def create_chat_history(conversation_history: list) -> str:
    """
    Creates a string chat history to pass onto the prompt for LLM to look at
    when rewriting user query for vector DB retrieval.

    Parameters:
        conversation_history (list) - An array of users conversation history with chatbot.
    
    Returns:
        chat_history (str) - Users chat history 
    """

    chat_history = ""

    for query in conversation_history:
        chat_history += query['role'] + ": " + query['content'] + '\n'

    return chat_history 


async def rag_pipeline(user_query: str, rag_user_query: str, urgency: str, conversation_history: list, first_turn: bool) -> str:
    """
    Passes user query and conversation through RAG pipeline.
    Parameters:
      user_query (str) - Users current question.
      rag_user_query (str) - Users query for RAG retrieval
      urgency (str) - Level of urgency 
      conversation_history (array) - An array of users conversation history with chatbot.
      first_turn (boolean) - True if this is user's first message, False otherwise.
    
    Returns:
      response (str) - Model's final response to users query based on history, and RAG data.
    """

    # Retrieve the closest response to users query 
    retrieved_chunk = search(rag_user_query)
    
    # Generate prompt to pass into LLM 
    prompt = augment_prompt(conversation_history, retrieved_chunk, user_query, urgency, first_turn)

    print(f"Prompt: {prompt}")

    
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


    #return {"response": "Model will give response shortly!"}
    