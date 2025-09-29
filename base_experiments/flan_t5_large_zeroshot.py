import pandas as pd
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, pipeline
)
import torch
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import gc

# -----------------------
print("----- START -----")  
# Constants block 
MODEL_NAME = "google/flan-t5-large"  
TEST_PATH = "/home/joberant/NLP_2425b/maozhaim/datasets/test_dataset.csv"    
TRAIN_PATH = "/home/joberant/NLP_2425b/maozhaim/datasets/train_dataset.csv"  
OUTPUT_DIR = "/home/joberant/NLP_2425b/inbalhasar/inbal_NLP/flan_t5_large/results" 
MODEL_SAVE_PATH = "/home/joberant/NLP_2425b/inbalhasar/inbal_NLP/flan_t5_large-finetuned-ai-human"  
LORA_DIR = "/home/joberant/NLP_2425b/inbalhasar/inbal_NLP/flan_t5_large/lora"  
MAX_LEN = 256  
EPOCHS = 2     
LEARNING_RATE = 1e-4  # keep LoRA-friendly LR
# -----------------------

# --- Clean Torch caches ---
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except AttributeError:
        pass

# -----------------------
print("--- GPU check ---")  
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print()
# -----------------------

# -----------------------
print("--- Model and tokenizer configurations ---") 
model_name = MODEL_NAME 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

# align label mapping order with base (0: Human, 1: AI)
id2label = {0: "Human", 1: "AI"} 
label2id = {"Human": 0, "AI": 1} 

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,  
    label2id=label2id   
)

# LoRA config for T5 (attention projections) + keep the classifier trainable
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "k", "v", "o"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["classification_head"]
)

model = get_peft_model(base_model, lora_cfg)

# Needed with gradient checkpointing on encoder-decoder models
model.enable_input_require_grads()
model.config.use_cache = False
model.gradient_checkpointing_enable()

# (Optional sanity check in logs)
try:
    model.print_trainable_parameters()
except Exception:
    pass

model.to(device)
print()
# -----------------------

# -----------------------
# Load test dataset FIRST & run zero-shot evaluation (match base flow)
# -----------------------
print("--- Load test dataset ---")  
print(f"Dataset path - {TEST_PATH}")  
df_test = pd.read_csv(TEST_PATH)  

# Ensure columns exist (reuse your flexible renaming)
rename = {}  
if "text" not in df_test.columns:
    if "content" in df_test.columns: rename["content"] = "text"
    elif "prompt" in df_test.columns: rename["prompt"] = "text"
if "label" not in df_test.columns:
    if "target" in df_test.columns: rename["target"] = "label"
if rename:
    df_test = df_test.rename(columns=rename)  

# Normalize labels (reuse helper)
def to_int_label(x: object) -> int:  
    s = str(x).strip().lower()
    if s in {"1", "1.0", "ai"}: return 1
    if s in {"0", "0.0", "human"}: return 0
    return int(float(s))

df_test["label"] = df_test["label"].apply(to_int_label)  

test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))  
print(f"Test samples - {len(test_dataset)}")  
texts = [ex["text"] for ex in test_dataset]              
true_labels = [int(ex["label"]) for ex in test_dataset]  
print()

print("--- Zero-shot testing ---")  
pre_pipe = pipeline(
    "text-classification",
    model=model,  
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN
)
print("Calculating zero-shot results") 
pre_results = pre_pipe(texts, batch_size=4)  
pre_pred_labels = [1 if r["label"] in ("AI", "LABEL_1", "1") else 0 for r in pre_results]  
pre_accuracy = accuracy_score(true_labels, pre_pred_labels) 
pre_precision = precision_score(true_labels, pre_pred_labels) 
print(f"Accuracy: {pre_accuracy:.4f}") 
print(f"Precision: {pre_precision:.4f}") 
print()
# --- collect misclassified zero-shot examples (LoRA) ---
mis_zero = []
for i, (t, y_true, r, y_pred) in enumerate(zip(texts, true_labels, pre_results, pre_pred_labels)):
    if y_pred != y_true:
        mis_zero.append({
            "index": i,
            "text": t,
            "true_label": id2label[y_true],
            "pred_label": id2label.get(y_pred, str(y_pred)),
            "raw_pred_label": r.get("label"),
            "score": float(r.get("score", float("nan")))
        })

print(f"Misclassified (zero-shot): {len(mis_zero)} / {len(true_labels)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
pd.DataFrame(mis_zero).to_csv(f"{OUTPUT_DIR}/misclassified_zero_shot.csv", index=False)

print("----- FINISH -----") 
