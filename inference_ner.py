import pandas as pd
test_df = pd.read_csv("/kaggle/input/actual-test-ner/test_NERLens with labels post annotator2 (3).csv")

import ast
import bisect
import re
from gliner import  GLiNER
labels = ["LAW", "VIOLATED BY", "VIOLATED ON","VIOLATION" ]
labels =  [l.lower() for l in  labels ]
wanted_labels =  labels

model = GLiNER.from_pretrained("bordias/gliner_base_legallens_ner",load_tokenizer =True)
model.cuda()

test_df.to_csv("test.csv")

def tokenize_with_offsets(text):
    """Dummy tokenizer.
    Use any tokenizer you want as long it as the same API."""
    tokens,starts,ends = zip(*[(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', text)])
    return tokens, starts, ends

def get_labels(starts, ends, spans):
    """Convert offsets to sequence labels in BIO format."""
    labels = ["O"]*len(starts)
    spans = sorted(spans)
#    print(spans)
    for s,e,l in spans:
        li = bisect.bisect_left(starts, s)
        ri = bisect.bisect_left(starts, e)
        ni = len(labels[li:ri])
#        print(li, ri, ni)
        labels[li] = f"B-{l}"
        labels[li+1:ri] = [f"I-{l}"]*(ni-1)
    return labels

def filter_ents(ents):

    wanted_dict = {k: [] for k in wanted_labels}

    for ent in ents:

        if len(wanted_dict[ent['label']])>0:
            if wanted_dict[ent['label']]['score']<ent['score']:
                wanted_dict[ent['label']]=ent

        else:
            wanted_dict[ent['label']] = ent

    return [wanted_dict[k] for k in wanted_dict if len(wanted_dict[k] )>0]

ner_tags=[]
for i, r in test_df.iterrows():

    arr = ast.literal_eval(r["tokens"])

    text = ' '.join([t for t in arr])
    ents = model.predict_entities(text, wanted_labels   , threshold =0.5)
    ents =filter_ents(ents)

    (tokens, starts, ends) = tokenize_with_offsets(text)
    spans = [(ent['start'],ent['end'],ent["label"].upper()) for ent in ents]


    labels=get_labels(starts, ends, spans)
    assert(len(labels)==len(arr))
    ner_tags.append(labels)
    
test_df['ner_tags_predictions'] =ner_tags
test_df.to_csv('predictions_NERLens.csv' )

def check_ner_format(predictions_file_path, test_file_path):
    """
    Check the format of the NER prediction file.
    The file should be in CSV format with columns: id, tokens, ner_tags
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
    expected_columns = ['id', 'tokens', 'ner_tags']
    pred_columns = list(df.columns)
    for expected_col in expected_columns:
        if expected_col not in pred_columns:
            return False, f"Incorrect columns. Expected: {expected_columns}, Found: {pred_columns}"
    
    # Check number of rows
    expected_ner_num_rows = len(test_df)
    predictions_ner_num_rows = len(df)
    if predictions_ner_num_rows != expected_ner_num_rows:
        return False, f"Incorrect number of predictions. Expected: {expected_ner_num_rows}, Found: {predictions_ner_num_rows}"

    return True, "NER prediction file format is correct."


check_ner_format('predictions_NERLens.csv' ,"test.csv")




import ast 

df =test_df
truths, preds, tokens = list(df["ner_tags"]),list(df["ner_tags_predictions"]), list(df["tokens"])
truths, tokens = [ast.literal_eval(t) for t in truths],[ast.literal_eval(pred) for pred in tokens]

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
y_true = truths
y_pred = preds

print(classification_report(y_true, y_pred,digits=4))
