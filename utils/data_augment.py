import pandas as pd 
import numpy as np 
import json 
from collections import defaultdict, deque
from pathlib import Path
from data_preprocess import * 
import random 

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

def _load_data():
    """Load the raw symptom CSV files used for augmentation."""

    dataset_df = pd.read_csv(REPO_ROOT / 'data' / 'symptom_data' / 'dataset.csv')
    symptom_description_df = pd.read_csv(REPO_ROOT / 'data' / 'symptom_data' / 'symptom_Description.csv')
    symptom_precaution_df = pd.read_csv(REPO_ROOT / 'data' / 'symptom_data' / 'symptom_precaution.csv')  
    symptom_severity_df = pd.read_csv(REPO_ROOT / 'data' / 'symptom_data' / 'Symptom-severity.csv')

    return dataset_df, symptom_description_df, symptom_precaution_df, symptom_severity_df 

def _preprocess(dataset_df, symptom_description_df, symptom_precaution_df, symptom_severity_df):
    """Preprocess the loaded symptom data into lookup-friendly structures."""

    # Preprocess the data
    dataset_df = dataset_df.drop_duplicates()
    dataset_df, symptom_description_df, symptom_precaution_df, symptom_severity_df = process_datasets(
        dataset_df, symptom_description_df, symptom_precaution_df, symptom_severity_df
    )
    disease_set = set(dataset_df['Disease'])
    dataset = remove_missing(dataset_df, disease_set)
    symptom_precaution_df = remove_missing_precaution(symptom_precaution_df)

    return dataset, symptom_description_df, symptom_precaution_df, symptom_severity_df, disease_set

def augment():
    """Augment the symptom dataset with additional synthetic symptom combinations."""

    disease_to_symptoms = defaultdict(set)
    augmented = deque()

    symptom_dataset_path = REPO_ROOT / 'JSON_data' / 'symptom_dataset.json'

    with open(symptom_dataset_path, 'r') as file:
        symptom_dataset = json.load(file)

    dataset_df, symptom_description_df, symptom_precaution_df, symptom_severity_df = _load_data()

    dataset, symptom_description_df, symptom_precaution_df, symptom_severity_df, disease_set = _preprocess(
        dataset_df, symptom_description_df, symptom_precaution_df, symptom_severity_df
    )
    
    for disease, symptoms_list in dataset.items():
        for symptom in symptoms_list:
            for symp in symptom:
                symp = symp.strip().replace('_', ' ').lower()
                disease_to_symptoms[disease].add(symp)
    
    description_lookup = {
        row['Disease']: row['Description']
        for _, row in symptom_description_df.iterrows()
    }

    precaution_lookup = {
        row['Disease']: list(row.iloc[1:].dropna())
        for _, row in symptom_precaution_df.iterrows()
    }
    
    for disease in disease_to_symptoms:
        for _ in range(200):
            augmented.append(_generate_case(disease, disease_to_symptoms[disease], description_lookup, precaution_lookup))
    
    augmented = list(augmented)

    with open(symptom_dataset_path, 'w') as file:
        json.dump(symptom_dataset + augmented, file, indent=4)
        

def _generate_case(disease, symptoms, description_lookup, precaution_lookup, min_symptoms=3, max_symptoms=6):
    """Generate one synthetic symptom case for a disease."""
    symptoms = list(symptoms)
    if len(symptoms) <= 0:
        return None

    max_k = min(max_symptoms, len(symptoms))
    min_k = min(min_symptoms, max_k)

    k = random.randint(min_k, max_k)
    sample = random.sample(list(symptoms), k=k)

    data = {
        "disease": disease, 
        "description": description_lookup[disease],
        "symptoms": sample, 
        "precaution": precaution_lookup[disease]
    }

    return data 




