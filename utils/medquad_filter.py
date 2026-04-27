import re 

RESOURCE_PATTERNS =[
    r'genetic testing registry',
    r'gene review',
    r'these resources from medlineplus',
    r'medlineplus offer information about the diagnosis and management',
    r'find eye health organizations',
    r'you can use the medlineplus medical dictionary',
]

CONTACT_PATTERNS = [
    r'national eye institute',
    r'national institutes of health',
    r'u.s. department of health and human services',
    r'2020 vision place',
    r'bethesda, md',
    r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    r'\b(?:https?://|www\.)\S+\b',
    r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b'
]

QUESTION_ECHO_PATTERNS = [
    r'what causes',
    r'what are the symptoms',
    r'what are the treatments',
    r'who is at risk',
    r'how to prevent',
    r'how to diagnose',
]

def _normalize_spaces(text):
    if text is None:
        return ''
    return re.sub(r'\s+', ' ', str(text)).strip()

def _classify_answer(answer):
    """
    Classify why a row is noisy based on the question and it's answer.
    Parameters:
     - answer (str): The answer to the question.
    """
    noise_reasons = []
    if any(re.search(pattern, answer, re.IGNORECASE) for pattern in RESOURCE_PATTERNS):
        noise_reasons.append('resource')
    
    if any(re.search(pattern, answer, re.IGNORECASE) for pattern in CONTACT_PATTERNS):
        noise_reasons.append('contact')
    
    if any(re.search(pattern, answer, re.IGNORECASE) for pattern in QUESTION_ECHO_PATTERNS):
        noise_reasons.append('question_echo')

    return noise_reasons 
    


def _should_drop_row(answer):
    """
    Determine if a row should be dropped 
    Parameters:
     - answer (str): The answer to the question.
    Returns:
     - bool: True if should be dropped, False otherwise.
    """

    strong_contact_patterns = [
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
        r'\b(?:https?://|www\.)\S+\b',
        r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b'
    ]

    weak_contact_patterns = [
        r'national eye institute',
        r'national institutes of health',
        r'u.s. department of health and human services',
        r'2020 vision place',
        r'bethesda, md',
    ]

    noise_reasons = _classify_answer(answer)
    if 'question_echo' in noise_reasons:
        starts_with_question = any(answer.lower().startswith(pattern) for pattern in QUESTION_ECHO_PATTERNS)
        if starts_with_question:
            return True 
    
    if 'contact' in noise_reasons:
        strong_contact_count = 0
        weak_contact_count  = 0

        for pattern in strong_contact_patterns:
            strong_contact_count += len(re.findall(pattern, answer, re.IGNORECASE))
        
        for pattern in weak_contact_patterns:
            weak_contact_count += len(re.findall(pattern, answer, re.IGNORECASE))
        
        if strong_contact_count >= 2:
            return True 
        elif strong_contact_count > 0 and weak_contact_count > 0:
            return True 
        elif weak_contact_count > 3:
            return True 
    
    if 'resource' in noise_reasons:
        resource_count = sum(len(re.findall(pattern, answer, re.IGNORECASE)) for pattern in RESOURCE_PATTERNS)

        if resource_count > 1:
            return True 
    
    return False 


def filter_medquad_data(data):
    """
    Filter the MedQuAD dataset to remove any noisy rows.
    Parameters:
     - data: The MedQuAD dataset to be filtered.
    """

    # Remove any extra spaces and whitespace characters from the answers 
    data = data.copy()
    data['answer'] = data['answer'].apply(_normalize_spaces)

    data['should_drop'] = data.apply(lambda row: _should_drop_row(row['answer']), axis=1)
    return data[data['should_drop'] == False]
