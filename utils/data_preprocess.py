# Imports 
import pandas as pd 
import numpy as np 


def process_datasets(dataset_df, symptom_description_df, symptom_precaution_df, symptom_severity_df):
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

def remove_missing_precaution(symptom_precaution_df):
    # Replace NaN/missing column values with "no specific precautions" since there is only 2
    text = "no specific precautions"
    symptom_precaution_df.fillna({"Precaution_3": text, "Precaution_4": text}, inplace=True)
    return symptom_precaution_df

