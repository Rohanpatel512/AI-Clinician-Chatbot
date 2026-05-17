def augment_prompt(conversation_history, retrieved_chunk, user_query):
    """
    Creates an augmented prompt using conversation history, retrieved chunk
    and users current query.

    Parameters:
     conversation_history (str) - last 5 conversation exchanges
     retrieved_chunk (str) - Retrieved response for users current query.
     user_query (str) - Users current query.

    Return
     prompt (str) - Augmented prompt to send to LLM.
    """

    prompt = f"""
    You are a helpful medical assistant chatbot.

    Conversation History:
    {conversation_history}

    Relevant Medical Information:
    {retrieved_chunk}

    User Question:
    {user_query}

    Answer the user's question using the relevant medical information provided above.
    If the information is insufficient, say so clearly.
    """

    return prompt 