from nltk import sent_tokenize

sections = ["description", "symptoms", "when to see a doctor", "causes",
            "risk factors", "diagnosis", "treatment"]
def semantic_chunking(document, chunk_size=3000):
    """
    Performs semantic chunking on a document, splitting at logical text boundaries.
    Parameters:
        document (str) - The text document to process
    Returns:
        lst (List) - Semantically chunked documents with metadata. 
    """

    chunks = []

    for data in document:
        disease = data['disease']

        for section in sections:
            text = data.get(section)

            if not text:
                continue 

            sentences = sent_tokenize(text)

            current_chunk = []
            current_length = 0
            id = 0

            for sentence in sentences:
                sentence_len = len(sentence)

                if current_length + sentence_len > chunk_size:
                    if current_chunk:
                        chunks.append({
                            "id": f"{disease}_{section}_{id}",
                            "chunk_text": " ".join(current_chunk),
                            "disease": disease, 
                            "section": section
                        })

                    id += 1
                    current_chunk = [sentence]
                    current_length = sentence_len + 1
                
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_len 

            
            if current_chunk:
                chunks.append({
                    "id": f"{disease}_{section}_{id}",
                    "chunk_text": " ".join(current_chunk),
                    "disease": disease, 
                    "section": section
                })

            

    return chunks 