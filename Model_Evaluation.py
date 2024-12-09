'''
This code is for evaluate the model. Run it after finetuning.
'''

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from datasets import Dataset
import torch
from difflib import SequenceMatcher
import re

def load_custom_dataset(file_path):
    """load datas"""
    def read_jsonl(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return read_jsonl(file_path)


def preprocess_data(tokenizer, dataset, max_length=256):
    """preprocess raw data sets"""
    inputs = tokenizer([item["context"] for item in dataset],
                       max_length=max_length,
                       truncation=True,
                       padding="max_length",
                       return_tensors="pt")
    targets = [item["triplets"] for item in dataset]
    return inputs, targets


def clean_triplet_text(text):
    """clean marks of the labels"""
    text = re.sub(r"<(triplet|subj|obj)>", "", text).strip().lower()

    return text


def lcs_length(s1, s2):
    """
    compute lcs


    """
    m, n = len(s1), len(s2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
def compute_lcs_f1(predictions, targets):
    """
    Calculate overall precision, recall, and F1 score based on ROUGE-L,
    using the LCS for similarity.
    """
    total_matches = 0
    total_predicted = 0
    total_targets = 0

    for pred_triplet, target_triplet in zip(predictions, targets):
        cleaned_pred = clean_triplet_text(pred_triplet)
        cleaned_target = clean_triplet_text(target_triplet)

        if not cleaned_pred or not cleaned_target:
            continue

        # Calculate LCS length
        lcs_len = lcs_length(cleaned_pred, cleaned_target)

        # Update totals
        total_matches += lcs_len
        total_predicted += len(cleaned_pred)
        total_targets += len(cleaned_target)

    # Compute overall metrics
    precision = total_matches / total_predicted if total_predicted > 0 else 0
    recall = total_matches / total_targets if total_targets > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    # model path
    test_file = "./test_rebel.json"
    trained_model_path = "./rebel-finetuned"  # path of trained model
    local_rebel_model_path = "./rebel-large"  # path of model
    KG_model_path = "./ibm/knowgl-large"  # path of model
    # load datas
    test_data = load_custom_dataset(test_file)


    # load models
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_path)
    rebel_model = AutoModelForSeq2SeqLM.from_pretrained(local_rebel_model_path)
    KG_model = AutoModelForSeq2SeqLM.from_pretrained(KG_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    rebel_model.to(device)
    KG_model.to(device)
    # clean the data
    inputs, targets = preprocess_data(tokenizer, test_data)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # generate outputs
    trained_model_predictions = []
    rebel_model_predictions = []
    KG_model_predictions = []
    with torch.no_grad():
        for i in range(len(test_data)):

            trained_output = trained_model.generate(
                inputs["input_ids"][i].unsqueeze(0),
                max_length=20,
                num_beams=4,
                top_p=0.95,
                early_stopping=True
            )
            trained_pred = tokenizer.decode(trained_output[0], skip_special_tokens=True)
            trained_model_predictions.append(trained_pred)


            rebel_output = rebel_model.generate(
                inputs["input_ids"][i].unsqueeze(0),
                max_length=20,
                num_beams=4,
                top_p=0.95,
                early_stopping=True
            )
            rebel_pred = tokenizer.decode(rebel_output[0], skip_special_tokens=True)
            rebel_model_predictions.append(rebel_pred)

        for i in range(len(test_data)):

            KG_output = KG_model.generate(
                inputs["input_ids"][i].unsqueeze(0),
                max_length=20,
                num_beams=4,
                top_p=0.95,
                early_stopping=True
            )
            KG_pred = tokenizer.decode(KG_output[0], skip_special_tokens=True)
            KG_model_predictions.append(KG_pred)
    # compute F1
    f1_trained = compute_lcs_f1(trained_model_predictions, targets)
    f1_rebel = compute_lcs_f1(rebel_model_predictions, targets)
    f1_KG = compute_lcs_f1(KG_model_predictions, targets)
    # outputs
    print(f"Trained")
    print(f"Precision: {f1_trained['precision'] * 100:.2f}%, Recall: {f1_trained['recall'] * 100:.2f}%, F1: {f1_trained['f1'] * 100:.2f}%")

    print(f" Rebel")
    print(f"Precision: {f1_rebel['precision'] * 100:.2f}%, Recall: {f1_rebel['recall'] * 100:.2f}%, F1: {f1_rebel['f1'] * 100:.2f}%")

    print(f"Knowgl")
    print(f"Precision: {f1_KG['precision'] * 100:.2f}%, Recall: {f1_KG['recall'] * 100:.2f}%, F1: {f1_KG['f1'] * 100:.2f}%")

if __name__ == "__main__":
    main()
