# Imports 
import pandas as pd 
import numpy as np 
import json 

def process_datasets(dataset_df, symptom_description_df, symptom_precaution_df, symptom_severity_df):
    """
    Normalize text columns across the symptom data tables.
    Parameters:
     - dataset_df: Table of disease and their symptoms (Pandas DataFrame)
     - symptom_description_df: Table of diseases and their descriptions (Pandas DataFrame)
     - symptom_precaution_df: Table of precautions for each disease (Pandas DataFrame)
     - symptom_severity_df: Table of severity for each symptom (Pandas DataFrame)
    
    Returns:
     - dataset_df
     - symptom_description_df
     - symptom_precaution_df
     - symptom_severity_df
    """
    # Remove whitespace and underscoring from each column in each table 
    dataset_cols = dataset_df.select_dtypes(object).columns
    description_cols = symptom_description_df.select_dtypes(object).columns
    precaution_cols = symptom_precaution_df.select_dtypes(object).columns
    severity_cols = symptom_severity_df.select_dtypes(object).columns

    dataset_df[dataset_cols] = dataset_df[dataset_cols].apply(
        lambda x: x.str.strip().str.lower().str.replace('_', ' ')
    )
    symptom_description_df[description_cols] = symptom_description_df[description_cols].apply(
        lambda x: x.str.strip().str.lower().str.replace('_', ' ')
    )
    symptom_precaution_df[precaution_cols] = symptom_precaution_df[precaution_cols].apply(
        lambda x: x.str.strip().str.lower().str.replace('_', ' ')
    )
    symptom_severity_df[severity_cols] = symptom_severity_df[severity_cols].apply(
        lambda x: x.str.strip().str.lower().str.replace('_', ' ')
    )

    return dataset_df, symptom_description_df, symptom_precaution_df, symptom_severity_df

def remove_missing(dataset_df, disease_set):
    """
    Build a disease-to-symptom mapping while filtering missing symptom values.
    Parameters:
     - dataset_df: Table of disease and their symptoms. (Pandas DataFrame)
     - disease_set: A set of diseases. (Python set)
    
    Returns:
     - dataset: Dictionary of disease and different combinations of symptoms (Dictionary)
    """
    # Remove missing values from dataset.csv
    dataset = dict()

    for disease in disease_set:
        symptoms = dataset_df[dataset_df['Disease'] == disease].iloc[:, :-1].values
        dataset[disease] = []
        for symptom in symptoms:
            symptom = np.array(symptom[1:])

            cleaned = [
                s for s in symptom
                if (not pd.isna(s)) and (str(s).lower() != 'nan')
            ]

            dataset[disease].append(cleaned)
    
    return dataset

def detect_low_quality_precautions(symptom_precaution_df):
    """
    Detects any precautions that are vague or low-level
    Parameters:
     - symptom_precaution_df: A table of precautions for each disease (Pandas DataFrame)
    
    Returns:
     - low_quality: A set of low information/vague precautions. (Python set)
    """

    low_quality = set()
    STR_LENGTH = 3

    precautions = symptom_precaution_df.values[:, 1:]

    for row in precautions:
        for precaution in row: 
            if len(precaution.split()) <= 3:
                low_quality.add(precaution)
    
    return low_quality


def apply_precaution_mapping(symptom_precaution_df):
    """
    Applies new and more specific precautions to cells which contains vague ones.
    Parameters:
     - symptom_precaution_df: A table of precautions for each disease (Pandas DataFrame)
    
    Returns:
     - symptom_precaution_df
    """
    
    # Open up the mapping JSON file 
    with open('../data/mappings/precaution_mapping.json', 'r') as file:
        precaution_map = json.load(file)

    precaution_cols = symptom_precaution_df.columns[1:]

    for column in precaution_cols:
        symptom_precaution_df[column] = symptom_precaution_df[column].apply(
            lambda x: (
                precaution_map.get(x, x)
                if pd.notna(x)
                else x
            )

        )

    return symptom_precaution_df 
