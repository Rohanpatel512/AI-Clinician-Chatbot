# Import 
from transformers import pipeline 
import re 

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
THRESHOLD = 0.50

def analyze_routing(chat_history: str):
    """
    Given the chat history it checks for any red flag symptoms and then sees if the 
    intent of was just a general question or something they are actively dealing with.
    Helps model generate a more direct and safe response.

    Parameters:
        chat_history (str) - History of conversation up to most recent user query.
    
    Returns:
        inquiry_type - GENERAL if it is just a general question or urgency level if it is something user is actively dealing with.
    """

    INTENT_LABELS = [
        "the user is describing a current or recent health concern affecting themselves or another person",
        "the user is asking a general educational medical question without describing an actual health concern"
    ]

    concern = INTENT_LABELS[0]
    general = INTENT_LABELS[1]

    output = classifier(chat_history, INTENT_LABELS, multi_label=False)

    top_label = output["labels"][0]
    confidence = output["scores"][0]

    if top_label == general and confidence >= THRESHOLD:
        return "GENERAL"

    if top_label == concern and confidence >= THRESHOLD:
        urgency = classify_urgency(chat_history)

        if urgency == "symptoms suggesting a potentially life-threatening emergency requiring immediate emergency care":
            return "CRITICAL"
        elif urgency == "symptoms requiring prompt medical assessment but not clearly indicating an immediate emergency":
            return "ACUTE"
        elif urgency == "symptoms that can generally be managed with routine medical advice or non-urgent care":
            return "SUBACUTE"

    return "UNCERTAIN"


def classify_urgency(chat_history: str) -> str:
    """
    Classify the level of urgency based of conversation history up till users
    most recent query.

    Parameters:
        chat_history (str) - History of conversation up to most recent query. 
    
    Returns:
        urgency_level (str) - Level of urgency
    """

    URGENCY_LABELS = [
        "symptoms suggesting a potentially life-threatening emergency requiring immediate emergency care",
        "symptoms requiring prompt medical assessment but not clearly indicating an immediate emergency",
        "symptoms that can generally be managed with routine medical advice or non-urgent care"
    ]

    output = classifier(chat_history, URGENCY_LABELS, multi_label=False)

    top_label = output["labels"][0]

    return top_label
