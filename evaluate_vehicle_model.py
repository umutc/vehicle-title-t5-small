import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from main import prepare_data, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, device


MODEL_DIR = "./final_vehicle_model"
CSV_PATH = "dealershipvehicles_dataset.csv"
MAX_EVAL_SAMPLES = 20000
BATCH_SIZE = 128


def parse_output(text):
    parts = {}
    for segment in text.split("|"):
        segment = segment.strip()
        if ":" in segment:
            key, value = segment.split(":", 1)
            parts[key.strip().lower()] = value.strip()
    return parts


class EvalDataset(Dataset):
    def __init__(self, df):
        self.inputs = df["input_text"].tolist()
        self.targets = df["target_text"].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_text": self.inputs[idx], "target_text": self.targets[idx]}


def evaluate():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    df = prepare_data(CSV_PATH)
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)

    if MAX_EVAL_SAMPLES is not None and MAX_EVAL_SAMPLES < len(val_df):
        val_df = val_df.sample(MAX_EVAL_SAMPLES, random_state=42)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    dataset = EvalDataset(val_df)

    def collate_fn(batch):
        texts = [b["input_text"] for b in batch]
        targets = [b["target_text"] for b in batch]
        encodings = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        return encodings, targets

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    total = len(dataset)
    num_batches = len(dataloader)
    exact_match = 0
    year_correct = 0
    make_correct = 0
    model_correct = 0

    for batch_idx, (encodings, targets) in enumerate(dataloader, start=1):
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model.generate(**encodings, max_length=MAX_TARGET_LENGTH)

        for pred_ids, target_text in zip(outputs, targets):
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

            if pred_text == target_text:
                exact_match += 1

            true_parts = parse_output(target_text)
            pred_parts = parse_output(pred_text)

            if pred_parts.get("year") == true_parts.get("year"):
                year_correct += 1
            if pred_parts.get("make") == true_parts.get("make"):
                make_correct += 1
            if pred_parts.get("model") == true_parts.get("model"):
                model_correct += 1

        if batch_idx % 10 == 0 or batch_idx == num_batches:
            processed = min(batch_idx * BATCH_SIZE, total)
            percent = 100.0 * processed / total if total else 0.0
            print(f"Progress: {processed}/{total} samples ({percent:.1f}%)")

    def pct(x):
        return 100.0 * x / total if total else 0.0

    print(f"Evaluated samples: {total}")
    print(f"Exact match accuracy: {exact_match}/{total} ({pct(exact_match):.2f}%)")
    print(f"Year accuracy: {year_correct}/{total} ({pct(year_correct):.2f}%)")
    print(f"Make accuracy: {make_correct}/{total} ({pct(make_correct):.2f}%)")
    print(f"Model accuracy: {model_correct}/{total} ({pct(model_correct):.2f}%)")


if __name__ == "__main__":
    evaluate()
