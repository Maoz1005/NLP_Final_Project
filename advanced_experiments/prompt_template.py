import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
import torch
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split


print("----- START -----")


# --- GPU check ---
print("--- GPU check ---")
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print()


# --- Model & tokenizer (generative, zero-shot) ---
print("--- Model and tokenizer configurations ---")
model_name = "google-t5/t5-small"
print(f"Model - {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"   # truncate from the right
tokenizer.model_max_length = 512      # cap encoder input length

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name) # Some models (GPT-2 for example) use this library
model.to(device)
print()


# --- Load test dataset ---
TEST_CSV = "/vol/joberant_nobck/data/NLP_368307701_2425b/maozhaim/datasets/M4_test_dataset.csv"
df_test = pd.read_csv(TEST_CSV)
df_test = df_test.groupby("label", group_keys=False).apply(lambda x: x.sample(n=50000, random_state=42))
texts = df_test["text"].astype(str).tolist()
true_labels = df_test["label"].astype(int).tolist()
print(f"Test samples - {len(texts)}")
print()


# --- Helpers ---
def normalize_label(s: str) -> int:
    """Map generated text to {AI=1, Human=0} robustly."""
    s = (s or "").strip().lower()
    if s.startswith("ai"):
        return 1
    if s.startswith("human"):
        return 0
    if "human" in s:
        return 0
    if "ai" in s:
        return 1
    return 1  # conservative fallback (AI)


def extract_generated_text(item):
    """Handle both dict and [dict] shapes from the pipeline."""
    if isinstance(item, list):
        item = item[0]
    return item.get("generated_text", "")


gen = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)


def run_and_map(input_texts):
    outputs = gen(
        input_texts,
        batch_size=8,
        truncation=True,     # tokenizer enforces model_max_length on inputs
        max_new_tokens=2,    # generate only the label token(s)
        do_sample=False
    )
    return [normalize_label(extract_generated_text(o)) for o in outputs]


# =========================
# A) No-prompt baseline
# =========================
no_prompt_inputs = [f"Text:\n{t}\n\nAnswer:" for t in texts]
print("Calculating no-prompt results")
no_labels = run_and_map(no_prompt_inputs)
acc = accuracy_score(true_labels, no_labels)
prec = precision_score(true_labels, no_labels, zero_division=0)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")


# =========================
# B) Prompt basline
# =========================
PROMPT_PREFIX = (
    "Classify the following textand decide whether it was written by an AI or a Human. "
    "Answer with exactly one word: AI or Human.\n\nText:\n"
)
prompt_inputs = [f"{PROMPT_PREFIX}{t}\n\nAnswer:" for t in texts]
print("Calculating 'prompted' results")
pred_labels = run_and_map(prompt_inputs)
acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels, zero_division=0)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")


print("----- FINISH -----")



