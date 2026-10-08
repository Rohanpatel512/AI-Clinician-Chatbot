# Imports 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import nltk
from nltk.tokenize import sent_tokenize

model_name = "Qwen/Qwen2.5-3B-Instruct"

# Load the model and tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token 
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left" 

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

model.config.pad_token_id = tokenizer.eos_token_id


def build_output_prompt(input_text, output_text):
    system_prompt = """You are rewriting medical assistant answers into a SAFE clinical assistant style.
    You are a clinical writing assistant.

    Rewrite medical assistant responses to sound natural, calm, and conversational while preserving all medical facts and the original level of urgency.

    Rules:
    - Ensure the rewritten response directly addresses the patient's question.
    - Ensure the rewritten response directly addresses the patient's question using only information supported by the original response.
    - Preserve the original medical meaning, facts, and level of urgency exactly.
    - Do not add, remove, diagnose, prescribe, or introduce new advice.
    - Rewrite only the response. Do not mention the rewriting task, instructions, or ask for the original text.
    - Return ONLY the rewritten medical response.
    """

    user_prompt = f"""
    Patient question:
    {input_text}

    medical response:
    {output_text}

    Rewrite the medical response:
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    return messages
    

    

def build_input_prompt(text):
    system_prompt = """You are rewriting a medical question to sound like a natural patient speaking to a doctor or medical chatbot.

    Rules:
    - Remove redundant repetition while preserving all unique symptoms, details, and medical meaning.
    - Return only ONE rewritten question. Do not include labels, explanations, or multiple versions.
    - Preserve the exact medical meaning. Do not add, remove, or change symptoms, emotions, or details.
    - Rewrite the question to sound like a natural patient describing their concern.
    - Keep the wording concise, conversational, and avoid exaggerated emotional tone.
    """

    input_text = f"""{text}"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": input_text
        }
    ]

    return messages 
    

def rewrite_text(prompts, mini_batch_size, metadata):
    results = []

    for start in range(0, len(prompts), mini_batch_size):

        # Batch the prompts 
        batch = prompts[start: start + mini_batch_size]

        # Tokenize each message in the batch 
        batch_text = [
            tokenizer.apply_chat_template(
                message, 
                tokenize=False, 
                add_generation_prompt=True
            )
            for message in batch
        ]

        inputs = tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )


        inputs = {k: v.to(model.device) for k, v in inputs.items()}    

        # Rewrite the text 
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

        generated = outputs[:, inputs["input_ids"].shape[1]:]

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        for idx, text in enumerate(decoded):

            original_index = start + idx

            cleaned = text.strip().strip("'").strip('"')

            if (
                "provide the medical assistant response" in cleaned.lower()
                or "rewrite the following" in cleaned.lower()
                or "rewrite this" in cleaned.lower()
            ):
                print("\n" + "="*80)
                print("FAILED ROW:", original_index)
                print("="*80)

                if metadata:
                    print("\nORIGINAL INPUT:")
                    print(metadata[original_index]["input"])

                    print("\nORIGINAL OUTPUT:")
                    print(metadata[original_index]["output"])

                print("\nPROMPT SENT:")
                print(batch[idx])

                print("\nMODEL RESPONSE:")
                print(cleaned)


        results.extend([text.strip() for text in decoded])

    return results
            
        