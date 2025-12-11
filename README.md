---
language: en
license: cc-by-nc-4.0
tags:
  - t5
  - t5-small
  - text2text-generation
  - vehicle-parsing
  - sequence-to-sequence
  - fine-tuned
pipeline_tag: text2text-generation
library_name: transformers
datasets:
  - private
model-index:
  - name: vehicle-title-t5-small
    results:
      - task:
          type: text2text-generation
          name: Vehicle Information Extraction
        metrics:
          - name: Exact Match
            type: exact_match
            value: 55.53
          - name: Year Accuracy
            type: accuracy
            value: 98.06
          - name: Make Accuracy
            type: accuracy
            value: 98.78
          - name: Model Accuracy
            type: accuracy
            value: 56.49
---

# Vehicle Title to Year/Make/Model (T5-small)

[![Model](https://img.shields.io/badge/Model-t5--small-blue)](https://huggingface.co/google-t5/t5-small)
[![License](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Transformers](https://img.shields.io/badge/Transformers-4.x-orange)](https://huggingface.co/docs/transformers)

A fine-tuned `t5-small` sequence-to-sequence model that extracts structured vehicle information (year, make, model) from free-form vehicle listing titles.

## Model Overview

| Property | Value |
|----------|-------|
| Base Model | `google-t5/t5-small` |
| Parameters | ~60 million |
| Model Size | ~231 MB (safetensors) |
| Task | Text-to-text generation |
| Input | Raw vehicle title with instruction prefix |
| Output | Structured string: `Year: XXXX \| Make: XXXX \| Model: XXXX` |

### Input/Output Format

**Input:**
```
extract vehicle info: New 2023 Yamaha YZ450F Monster Energy Edition
```

**Output:**
```
Year: 2023 | Make: Yamaha | Model: YZ450F
```

## Intended Use

- Normalizing marketplace or dealership listing titles into structured fields
- Preprocessing step for downstream analytics or search systems expecting `(year, make, model)` tuples
- Batch processing of vehicle inventory data

This model targets vehicle title parsing; it is not a general-purpose language model.

## Training Details

### Data

| Property | Value |
|----------|-------|
| Source | Private dealership inventory export |
| Total Samples | ~200,000 |
| Train/Val Split | 90% / 10% |
| Status | Proprietary; not released |

Fields used from source data:
- `onlineTitle` (free-text input)
- `catalogYear`, `catalogMake`, `catalogModel` (target labels)

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Batch Size | 32 |
| Learning Rate | 2e-4 |
| Weight Decay | 0.01 |
| Max Input Length | 256 tokens |
| Max Target Length | 128 tokens |
| Optimizer | AdamW (via Seq2SeqTrainer) |
| Gradient Checkpointing | Enabled |

### Hardware

| Component | Specification |
|-----------|---------------|
| Device | Apple MacBook Pro |
| Chip | Apple M1 Max |
| Backend | PyTorch MPS |
| Training Time | ~5 hours |

### Software Environment

- Python 3.14
- PyTorch 2.9.1
- Transformers < 4.57
- scikit-learn

## Evaluation Results

Evaluated on 19,793 held-out validation samples from the private corpus.

| Metric | Score |
|--------|-------|
| Exact String Match | 55.53% |
| Year Accuracy | 98.06% |
| Make Accuracy | 98.78% |
| Model Accuracy | 56.49% |

Field-wise accuracies are computed by parsing the generated string and comparing each field independently against ground truth.

## How to Use

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_ID = "umut-celik/vehicle-title-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
model.eval()

def predict_vehicle(title: str) -> str:
    input_text = "extract vehicle info: " + title
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
title = "New 2023 Yamaha YZ450F Monster Energy Edition"
print(predict_vehicle(title))
# Output: Year: 2023 | Make: Yamaha | Model: YZ450F
```

### Parsing the Output

```python
def parse_prediction(text: str) -> dict:
    parts = {}
    for segment in text.split("|"):
        segment = segment.strip()
        if ":" in segment:
            key, value = segment.split(":", 1)
            parts[key.strip().lower()] = value.strip()
    return parts

result = parse_prediction("Year: 2023 | Make: Yamaha | Model: YZ450F")
# {'year': '2023', 'make': 'Yamaha', 'model': 'YZ450F'}
```

## Limitations

- Trained on a single private dealership corpus; may overfit to specific brands, naming conventions, and title styles present in that data
- Titles deviating from training patterns (non-English text, very short titles, missing year information) may produce incorrect or incomplete outputs
- Model accuracy for the `model` field is lower than `year` and `make`; complex or uncommon model names are harder to extract
- Not robust against adversarial inputs; treat outputs as probabilistic predictions rather than validated facts

## License

This model is released under the [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. Commercial use is not permitted without explicit authorization.

## Citation

```bibtex
@misc{celik2024vehicletitle,
  author       = {Celik, Umut},
  title        = {Vehicle Title to Year/Make/Model Extraction with T5-small},
  year         = {2024},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/umut-celik/vehicle-title-t5-small}}
}
```

## Contact

- **Author:** Umut Celik
- **Email:** umut.celik@cix.csi.cuny.edu
- **GitHub:** [github.com/umutc](https://github.com/umutc)
- **Affiliation:** CUNY College of Staten Island
