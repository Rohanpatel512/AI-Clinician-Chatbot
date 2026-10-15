def augment_prompt(conversation_history: dict, retrieved_chunk: str, user_query) -> str:
    """
    Creates an augmented prompt using conversation history, retrieved chunk
    and users current query.

    Parameters:
     conversation_history (list of dictionaries) - Up to last 5 conversation exchanges
     retrieved_chunk (str) - Retrieved response for users current query.
     user_query (str) - Users current query.

    Return
     prompt (str) - Augmented prompt to send to LLM.
    """

    instruction = """
    Answer the patient's question directly with specific, sufficiently detailed, 
    patient-friendly guidance that explains why recommended actions may help. 
    Use relevant medical context and conversation history when helpful, and general medical knowledge when appropriate; 
    ignore unrelated context. Acknowledge uncertainty, avoid unsupported diagnoses, and ask focused follow-up questions when useful. 
    Prioritize medical safety and clearly direct patients to emergency services for possible medical emergencies.
    """

    prompt = (
        f"Instruction: {instruction}"
        f"Conversation history: {_format_conversation(conversation_history)}"
        f"Medical Context: {retrieved_chunk}"
        f"Patient: {user_query}"
    )

    return prompt 

def _format_conversation(conversation_history: list) -> str:
    """
    Formats the conversation from list to a readable dialouge
    between user and bot.

    Parameters:
     conversation_history (list of dictionaries) - Up to last 5 conversation exchanges
    
    Return
     dialogue (str) - Readable dialogue between user and bot.
    """
    dialogue = ""
    for query in conversation_history:
        dialogue += query["role"] + ": " + query["content"] + "\n"

    return dialogue


