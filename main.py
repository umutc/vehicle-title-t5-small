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

# --- 1. AYARLAR & KONFİGÜRASYON ---
MODEL_NAME = "t5-small"   # Hızlı eğitim için 'small'. Daha yüksek başarı için 't5-base' kullanabilirsiniz.
MAX_INPUT_LENGTH = 256    # onlineTitle için maksimum uzunluk
MAX_TARGET_LENGTH = 128   # Çıktı (Marka/Model/Yıl) için maksimum uzunluk
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-4
SAMPLE_SIZE = None  # Hızlı deneme yapmak için veriyi sınırlayalım. Tüm veri için None yapın.

CPU_THREADS = os.cpu_count() or 1

# CPU tarafını ve matmul hassasiyetini üst sınıra çekiyoruz
torch.set_num_threads(CPU_THREADS)
torch.set_float32_matmul_precision("high")

# Cihaz seçimi (GPU varsa kullan, yoksa CPU)
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

print(f"Kullanılan Cihaz: {device}")

# --- 2. VERİ HAZIRLIĞI ---
def prepare_data(filepath):
    """
    CSV dosyasını okur ve T5 modeli için Input-Target formatına getirir.
    """
    df = pd.read_csv(filepath)

    # Veri tiplerini string'e çevirelim (özellikle Yıl sütunu int gelebilir)
    df['catalogYear'] = df['catalogYear'].astype(str)
    df['catalogMake'] = df['catalogMake'].astype(str)
    df['catalogModel'] = df['catalogModel'].astype(str)
    df['onlineTitle'] = df['onlineTitle'].astype(str)

    # Input: Modelin ne yapacağını anlaması için bir prefix (önek) ekleyebiliriz.
    # Örnek Input: "extract vehicle info: 2022 Harley-Davidson FLTRX"
    df['input_text'] = "extract vehicle info: " + df['onlineTitle']

    # Target: Modelin üretmesini istediğimiz ideal çıktı formatı.
    # Örnek Target: "Year: 2022 | Make: Harley-Davidson | Model: FLTRX"
    # Ayıraç olarak " | " kullanıyoruz, sonradan parse etmek kolay olsun diye.
    df['target_text'] = (
        "Year: " + df['catalogYear'] +
        " | Make: " + df['catalogMake'] +
        " | Model: " + df['catalogModel']
    )

    return df[['input_text', 'target_text']]

# --- 3. DATASET SINIFI ---
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

        # Değişken uzunluklar için padding işini collator'a bırak; MPS'te gereksiz kopyaları azalt
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

# --- 4. EĞİTİM AKIŞI ---
def main():
    # 4.1 Veriyi Yükle
    print("Veri yükleniyor...")
    df = prepare_data('dealershipvehicles_dataset.csv')

    # Hızlı test için örneklem al (Prodüksiyon için bu satırı kaldırın)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        print(f"Hızlı test için {SAMPLE_SIZE} satır kullanılıyor...")
        df = df.sample(SAMPLE_SIZE, random_state=42)

    # Train/Validation Split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Eğitim Seti: {len(train_df)}, Doğrulama Seti: {len(val_df)}")

    # 4.2 Tokenizer ve Model Yükle
    print("Model ve Tokenizer yükleniyor...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.use_cache = False  # gradient checkpointing ile uyumlu hale getir
    model.to(device)

    # 4.3 Dataset Oluştur
    train_dataset = VehicleDataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    val_dataset = VehicleDataset(val_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    # 4.4 Training Argümanları
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
        fp16=(device.type == "cuda"), # Sadece CUDA için yarı hassasiyet
        bf16=False,  # MPS ve çoğu CPU için henüz güvenilir değil
        logging_steps=50,
        report_to="none", # WandB vb. kullanmıyorsanız
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=PIN_MEMORY,
        dataloader_prefetch_factor=PREFETCH_FACTOR,
        gradient_checkpointing=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4.5 Trainer Başlat
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    # Eğitimi Başlat
    print("Eğitim başlıyor...")
    trainer.train()
    model.to(device)  # Trainer, modeli kendi cihazına taşıyor; yeniden hizala

    # 4.6 Modeli Kaydet
    print("Model kaydediliyor...")
    model.save_pretrained("./final_vehicle_model")
    tokenizer.save_pretrained("./final_vehicle_model")

    # --- 5. TAHMİN (INFERENCE) ÖRNEĞİ ---
    print("\n--- TEST / INFERENCE ÖRNEĞİ ---")

    def predict_vehicle(text):
        input_text = "extract vehicle info: " + text
        inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(inputs, max_length=MAX_TARGET_LENGTH)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Örnek bir veri ile test edelim
    test_title = "New 2023 Yamaha YZ450F Monster Energy Edition"
    print(f"Girdi: {test_title}")
    prediction = predict_vehicle(test_title)
    print(f"Tahmin: {prediction}")

    # Veri setinden rastgele bir tane daha
    random_row = val_df.sample(1).iloc[0]
    real_input = random_row['input_text'].replace("extract vehicle info: ", "")
    print(f"\nGerçek Veriden Girdi: {real_input}")
    print(f"Beklenen (Hedef): {random_row['target_text']}")
    print(f"Model Tahmini: {predict_vehicle(real_input)}")

if __name__ == "__main__":
    main()
