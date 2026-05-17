from chunking import semantic_chunking
import xml.etree.ElementTree as ET
import pandas as pd 
import numpy as np 
from pathlib import Path
import json 

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATASET_URL = "hf://datasets/harmesh95/healthcare-disease-knowledge/healthcare_disease_dataset.csv"


def clean_data():
    """
    Cleans and preprocesses data to get it ready for chunking 
    and embedding.
    Parameters:
      None

    Returns:
      None
    """

    rag_data = _gather_disease_knowledge()
    chunks = semantic_chunking(rag_data)
    return chunks 


def _gather_disease_knowledge():
    """
    Loads, cleans, and structures disease knowledge data for
    RAG pipeline 
    
    Returns:
      rag_data (list): A list of dictionaries where each dictionary
      contains structured disease information.
    """

    columns_removed = [
        "Unnamed: 0",
        "main_link",
        "Diagnosis_treatment_link",
        "Doctors_departments_link",
        "Preparing for your appointment",
        "updated",
        "Prevention",
        "Complications",
        "Coping and support",
        "Lifestyle and home remedies",
    ]

    rag_data = []

    df = pd.read_csv(DATASET_URL)
    df.drop(columns=columns_removed, axis=1, inplace=True)

    df.columns = df.columns.str.strip().str.lower()
    print(df.columns)
    
    df["disease"] = df["disease"].astype(str).str.strip().str.lower()
    df = df.set_index("disease")

    
    for disease, row in df.iterrows():

        rag_record = {
            'disease': disease,
            'description': _is_safe(row.get('overview')),
            'symptoms': _is_safe(row.get('symptoms')),
            'when to see a doctor': _is_safe(row.get('when to see a doctor')),
            'causes': _is_safe(row.get('causes')),
            'risk factors': _is_safe(row.get('risk factors')),
            'diagnosis': _is_safe(row.get('diagnosis')),
            'treatment': _is_safe(row.get('treatment'))
        }

        rag_data.append(rag_record)
    
    return rag_data
    


def _is_safe(text):
    """
    Ensures text is not NaN, None, or empty
    Parameters:
      text (str) - The text being checked.
    Returns:
      text if it isn't NaN, empty, or None otherwise it returns an empty string.
    """
    if text is None:
        return ""

    text = str(text).strip()

    if text.lower() in ["nan", "none", ""]:
        return ""

    return text

if __name__ == "__main__":

    chunks = clean_data()
    
    data_path = REPO_ROOT / 'data' / 'rag_data.json'

    with open(data_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2)
    