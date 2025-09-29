import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, pipeline
)
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os

# -----------------------
# Constants block 
# -----------------------
print("----- START -----") 
MODEL_NAME = "google/flan-t5-base"  
TEST_PATH = "/home/joberant/NLP_2425b/maozhaim/datasets/test_dataset.csv"  
TRAIN_PATH = "/home/joberant/NLP_2425b/maozhaim/datasets/train_dataset.csv"
OUTPUT_DIR = "/home/joberant/NLP_2425b/inbalhasar/inbal_NLP/flan_t5_base/results" 
MODEL_SAVE_PATH = "/home/joberant/NLP_2425b/inbalhasar/inbal_NLP/flan_t5_base-finetuned-ai-human" 
MAX_LEN = 256 
EPOCHS = 2    
LEARNING_RATE = 5e-6 

# -----------------------
# GPU check section
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
# Model & tokenizer config
# -----------------------
print("--- Model and tokenizer configurations ---")  
model_name = MODEL_NAME  
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right" 

# align call style
id2label = {0: "Human", 1: "AI"}  
label2id = {"Human": 0, "AI": 1}  

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,   
    label2id=label2id,   
    use_safetensors=True 
)
model.config.pad_token_id = tokenizer.pad_token_id  
model.to(device)
print()

# -----------------------
# Load test dataset (used later for post-finetune testing)
# -----------------------
print("--- Load test dataset ---") 
print(f"Dataset path - {TEST_PATH}") 
df_test = pd.read_csv(TEST_PATH) 
rename = {}
if "text" not in df_test.columns:
    if "content" in df_test.columns: rename["content"] = "text"
    elif "prompt" in df_test.columns: rename["prompt"] = "text"
if "label" not in df_test.columns:
    if "target" in df_test.columns: rename["target"] = "label"
if rename:
    df_test = df_test.rename(columns=rename) 

def to_int_label(x: object) -> int:  
    s = str(x).strip().lower()
    if s in {"1", "1.0", "ai"}: return 1
    if s in {"0", "0.0", "human"}: return 0
    return int(float(s))

df_test["label"] = df_test["label"].apply(to_int_label)  
test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))  
texts = [ex["text"] for ex in test_dataset]             
true_labels = [int(ex["label"]) for ex in test_dataset] 
print(f"Test samples - {len(test_dataset)}")
print()

# -----------------------
# Load training dataset, rename/normalize, split 80/20
# -----------------------
print("--- Load training dataset ---")  
print(f"Dataset path - {TRAIN_PATH}")  
df = pd.read_csv(TRAIN_PATH)           

rename = {}
if "text" not in df.columns:
    if "content" in df.columns: rename["content"] = "text"
    elif "prompt" in df.columns: rename["prompt"] = "text"
if "label" not in df.columns:
    if "target" in df.columns: rename["target"] = "label"
if rename:
    df = df.rename(columns=rename)  

df["label"] = df["label"].apply(to_int_label)  

df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))  
val_dataset   = Dataset.from_pandas(df_val.reset_index(drop=True))    
print(f"Train samples - {len(train_dataset)}")
print(f"Validation samples - {len(val_dataset)}")
print()

# -----------------------
# Tokenization block
# -----------------------
print("--- Tokenization ---")
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)  
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])  
print()

# -----------------------
print("--- Define fine-tuning arguments ---") 
fine_tuning_args = TrainingArguments(  
    output_dir=OUTPUT_DIR,           
    eval_strategy="epoch",            
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,      
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=4,    
    num_train_epochs=EPOCHS,         
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_dir=None,
    fp16=False,
    save_total_limit=2,
    push_to_hub=False
)
print()

# -----------------------
# Metrics block
# -----------------------
print("--- Define metrics computations ---") 
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
print()

# -----------------------
# Trainer
# -----------------------
print("--- Define fine-tuning trainer ---")  
fine_tuning_trainer = Trainer(  
    model=model,
    args=fine_tuning_args,        
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
print()

print("--- Fine-tuning on base model ---") 
fine_tuning_trainer.train() 
print("Finished")
print()

# -----------------------
# Save model
# -----------------------
print("--- Save model ---") 
fine_tuning_trainer.save_model(MODEL_SAVE_PATH) 
print("Model saved")
print()

# -----------------------
# Post-finetune testing mirrors friend’s pipeline usage
# -----------------------
print("--- Fine-tuned testing ---") 
fine_tuning_pipe = pipeline(
    "text-classification",
    model=MODEL_SAVE_PATH,  
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN
)

print("Calculating fine-tuning results") 
fine_tuning_results = fine_tuning_pipe(texts, batch_size=8)
fine_tuning_pred_labels = [1 if r["label"] in ("AI", "LABEL_1", "1") else 0 for r in fine_tuning_results] 
accuracy = accuracy_score(true_labels, fine_tuning_pred_labels)
precision = precision_score(true_labels, fine_tuning_pred_labels)
recall = recall_score(true_labels, fine_tuning_pred_labels)  
f1 = f1_score(true_labels, fine_tuning_pred_labels)         
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")   
print(f"F1: {f1:.4f}")           
print()

# --- collect misclassified finetuned examples ---
mis_ft = []
for i, (t, y_true, r, y_pred) in enumerate(zip(texts, true_labels, fine_tuning_results, fine_tuning_pred_labels)):
    if y_pred != y_true:
        mis_ft.append({
            "index": i,
            "text": t,
            "true_label": id2label[y_true],
            "pred_label": id2label.get(y_pred, str(y_pred)),
            "raw_pred_label": r.get("label"),
            "score": float(r.get("score", float("nan")))
        })

print(f"Misclassified (finetuned): {len(mis_ft)} / {len(true_labels)}")

# Save for later review
os.makedirs(OUTPUT_DIR, exist_ok=True)
pd.DataFrame(mis_ft).to_csv(f"{OUTPUT_DIR}/misclassified_finetuned.csv", index=False)


print("----- FINISH -----") 
