import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset

# --- 1. SETTINGS & CONFIGURATION ---
MODEL_NAME = "t5-small"   # Use 'small' for faster training. For higher accuracy, try 't5-base'.
MAX_INPUT_LENGTH = 256    # Maximum length for onlineTitle
MAX_TARGET_LENGTH = 128   # Maximum length for output (Make/Model/Year)
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-4
SAMPLE_SIZE = None  # Limit data for quick experiments. Set to None for full dataset.

CPU_THREADS = os.cpu_count() or 1

# Maximize CPU threads and matmul precision
torch.set_num_threads(CPU_THREADS)
torch.set_float32_matmul_precision("high")

# Device selection (use GPU if available, otherwise CPU)
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

if device.type == "mps":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PIN_MEMORY = False if device.type == "mps" else True
NUM_WORKERS = 0 if device.type == "mps" else max(1, CPU_THREADS - 1)
PREFETCH_FACTOR = None if NUM_WORKERS == 0 else 2

print(f"Using device: {device}")

# --- 2. DATA PREPARATION ---
def prepare_data(filepath):
    """
    Reads CSV file and formats it as Input-Target pairs for T5 model.
    """
    df = pd.read_csv(filepath)

    # Convert data types to string (especially Year column which may be int)
    df['catalogYear'] = df['catalogYear'].astype(str)
    df['catalogMake'] = df['catalogMake'].astype(str)
    df['catalogModel'] = df['catalogModel'].astype(str)
    df['onlineTitle'] = df['onlineTitle'].astype(str)

    # Input: Add a prefix so the model understands the task
    # Example Input: "extract vehicle info: 2022 Harley-Davidson FLTRX"
    df['input_text'] = "extract vehicle info: " + df['onlineTitle']

    # Target: The ideal output format we want the model to generate
    # Example Target: "Year: 2022 | Make: Harley-Davidson | Model: FLTRX"
    # Using " | " as delimiter for easy parsing later
    df['target_text'] = (
        "Year: " + df['catalogYear'] +
        " | Make: " + df['catalogMake'] +
        " | Model: " + df['catalogModel']
    )

    return df[['input_text', 'target_text']]

# --- 3. DATASET CLASS ---
class VehicleDataset(Dataset):
    def __init__(self, data, tokenizer, input_max_len, target_max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        source_text = data_row['input_text']
        target_text = data_row['target_text']

        # Let collator handle padding for variable lengths; reduces unnecessary copies on MPS
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.input_max_len,
            truncation=True,
            padding=False,
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.target_max_len,
            truncation=True,
            padding=False,
        )

        return {
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "labels": target_encoding["input_ids"],
        }

# --- 4. TRAINING FLOW ---
def main():
    # 4.1 Load Data
    print("Loading data...")
    df = prepare_data('dealershipvehicles_dataset.csv')

    # Sample for quick testing (remove this for production)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        print(f"Using {SAMPLE_SIZE} samples for quick testing...")
        df = df.sample(SAMPLE_SIZE, random_state=42)

    # Train/Validation Split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Training set: {len(train_df)}, Validation set: {len(val_df)}")

    # 4.2 Load Tokenizer and Model
    print("Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.use_cache = False  # Make compatible with gradient checkpointing
    model.to(device)

    # 4.3 Create Dataset
    train_dataset = VehicleDataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    val_dataset = VehicleDataset(val_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    # 4.4 Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./vehicle_model_results",
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=(device.type == "cuda"),  # Half precision for CUDA only
        bf16=False,  # Not yet reliable for MPS and most CPUs
        logging_steps=50,
        report_to="none",  # If not using WandB etc.
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=PIN_MEMORY,
        dataloader_prefetch_factor=PREFETCH_FACTOR,
        gradient_checkpointing=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4.5 Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    # Start Training
    print("Starting training...")
    trainer.train()
    model.to(device)  # Trainer moves model to its own device; realign

    # 4.6 Save Model
    print("Saving model...")
    model.save_pretrained("./final_vehicle_model")
    tokenizer.save_pretrained("./final_vehicle_model")

    # --- 5. INFERENCE EXAMPLE ---
    print("\n--- TEST / INFERENCE EXAMPLE ---")

    def predict_vehicle(text):
        input_text = "extract vehicle info: " + text
        inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(inputs, max_length=MAX_TARGET_LENGTH)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Test with a sample
    test_title = "New 2023 Yamaha YZ450F Monster Energy Edition"
    print(f"Input: {test_title}")
    prediction = predict_vehicle(test_title)
    print(f"Prediction: {prediction}")

    # Another random sample from the dataset
    random_row = val_df.sample(1).iloc[0]
    real_input = random_row['input_text'].replace("extract vehicle info: ", "")
    print(f"\nInput from actual data: {real_input}")
    print(f"Expected (Target): {random_row['target_text']}")
    print(f"Model prediction: {predict_vehicle(real_input)}")

if __name__ == "__main__":
    main()
