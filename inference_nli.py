
import os
import torch
import datasets

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import collections



test_path = "/kaggle/input/test-data/testset_NLI_LegalLens (1).xlsx"
test_data = pd.read_excel(test_path)

test_data.to_csv('test_NLILens.csv')


def get_train_nli_data(legal_type: str) -> pd.DataFrame:
    justice_lens_dataset = datasets.load_dataset("darrow-ai/LegalLensNLI") # This is a snippet from our dataset
    #justice_lens_dataset = load_from_disk("data/nli")
    df = (
        justice_lens_dataset["train"]
        .filter(lambda example: example["legal_act"] != legal_type)
        .to_pandas()
    )
    return df


def get_test_nli_data(legal_type: str) -> pd.DataFrame:
    justice_lens_dataset = datasets.load_dataset("darrow-ai/LegalLensNLI") # This is a snippet from our dataset

    df = (
        justice_lens_dataset["train"]
        .filter(lambda example: example["legal_act"] == legal_type)
        .to_pandas()
    )
    return df



model_name = "sileod/deberta-v3-small-tasksource-nli"

def predict_one_sample(data, p):
    label = p([dict(text=data['premise'], text_pair=data['hypothesis'])])[0]['label']
    return label



device ="cuda"

def create_local_pipeline(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device = device)
    return classifier

def predict_one(data, lp):
    label = lp([dict(text=data['premise'], text_pair=data['hypothesis'])])
    return label

def get_results(model_path):
    

    lp = create_local_pipeline(model_path)

    responses =[]
    confidence_score =[]
    for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        response = predict_one_sample(row,lp)
        
        responses.append(response)
        confidence_score.append(predict_one(row,lp))



    labels = df_test["label"].to_list()
    macro_f1 =f1_score(labels, responses, average="macro"),
    metrics = {
    "micro_f1": f1_score(labels, responses, average="micro"),
    "macro_f1": macro_f1,#f1_score(labels, responses, average="macro"),
    "micro_precision": precision_score(labels, responses, average="micro"),
    "micro_recall": recall_score(labels, responses, average="micro"),
    "macro_precision": precision_score(labels, responses, average="macro"),
    "macro_recall": recall_score(labels, responses, average="macro"),
    "accuracy": accuracy_score(labels, responses),
        }

    return metrics, confidence_score


model_paths = { "wage" :"bordias/deberta_v3_legallens_nli_wage"  ,
                "tcpa": "bordias/deberta_v3_legallens_nli_tcpa",
                "cp" :  "bordias/deberta_v3_legallens_nli_cp",
               "privacy" :  "bordias/deberta_v3_legallens_nli_privacy"}

pipe_line  = {} 

for mp in model_paths:

    pipe_line[mp] = create_local_pipeline(model_paths[mp])


def winner(p):
   return p[0]['score']


predictions =[]

for i,row in test_data.iterrows():
    preds =[]
    for k in model_paths:
        outcomes =predict_one(row,pipe_line[k])
        preds.append(outcomes)
    predictions.append(max(preds, key=winner)[0]['label'])
    
test_data['label']=predictions
test_data.to_csv('predictionsNLILens.csv')



def check_nli_format(predictions_file_path, test_file_path):
    """
    Check the format of the NLI prediction file.
    The file should be in CSV format with columns: Premise, hypothesis, label
    """
    try:
        df = pd.read_csv(predictions_file_path)
    except Exception as e:
        return False, f"Error reading predictions CSV file: {e}"
    
    try:
        test_df = pd.read_csv(test_file_path)
    except Exception as e:
        return False, f"Error reading test CSV file: {e}"
    
    # Check expected columns
    expected_columns = ['premise', 'hypothesis', 'label']
    pred_columns = list(df.columns)
    for expected_col in expected_columns:
        if expected_col not in pred_columns:
            return False, f"Incorrect columns. Expected: {expected_columns}, Found: {pred_columns}"
    
    # Check number of rows
    expected_nli_num_rows = len(test_df)
    predictions_nli_num_rows = len(df)
    if predictions_nli_num_rows != expected_nli_num_rows:
        return False, f"Incorrect number of predictions. Expected: {expected_nli_num_rows}, Found: {predictions_nli_num_rows}"
    
    return True, "NLI prediction file format is correct."



# Check NLI prediction file
nli_predictions_file_path = 'predictionsNLILens.csv' # replace with file path
nli_test_file_path = 'test_NLILens.csv' # replace with file path
is_valid, message = check_nli_format(nli_predictions_file_path, nli_test_file_path)
print(f"NLI File Check: {message}")
