import pandas as pd
import os

print("--- START ---")

print("--- Read CSVs ---")
df_1k = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/1k_rows.csv")
df_41k = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/41k_rows.csv")
df_487k = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/487k_rows.csv")
df_60_40 = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/essays-60_40.csv")
df_human = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/essays-mostly_human.csv")

print("--- Change names of columns ---")
df_1k = df_1k.rename(columns={"text_content": "text"})
df_487k = df_487k.rename(columns={"generated": "label"})
df_60_40 = df_60_40.rename(columns={"generated": "label"})
df_human = df_human.rename(columns={"generated": "label"})

print("--- Change labels to int ---")
df_1k['label'] = df_1k['label'].apply(lambda x: int(float(x)))
df_41k['label'] = df_41k['label'].apply(lambda x: int(float(x)))
df_487k['label'] = df_487k['label'].apply(lambda x: int(float(x)))
df_60_40['label'] = df_60_40['label'].apply(lambda x: int(float(x)))
df_human['label'] = df_human['label'].apply(lambda x: int(float(x)))

print("--- Keep only relevant columns ---")
df_1k = df_1k[['text', 'label']]
df_41k = df_41k[['text', 'label']]
df_487k = df_487k[['text', 'label']]
df_60_40 = df_60_40[['text', 'label']]
df_human = df_human[['text', 'label']]

print("--- Save original dataset name ---")
df_1k['original_dataset'] = "1k_rows"
df_41k['original_dataset'] = "41k_rows"
df_487k['original_dataset'] = "487k_rows"
df_60_40['original_dataset'] = "essays-60_40"
df_human['original_dataset'] = "essays-mostly_human"

print("--- Training-Test splitting ---")
# For small datasets (rows_nums < 10K) - 20%/80% (training/test)
# For big datasets (rows_num >= 10K) - train on 1K-2K examples from each label, the rest for test
df_1k_train = df_1k.groupby("label", group_keys=False).apply(lambda x: x.sample(n=135, random_state=42)) # 0.2 * 1367 ~ 270 -> 135 from each label
df_41k_train = df_41k.groupby("label", group_keys=False).apply(lambda x: x.sample(n=2000, random_state=42)) # 2000 from each label (41K+)
df_487k_train = df_487k.groupby("label", group_keys=False).apply(lambda x: x.sample(n=2000, random_state=42)) # 2000 from each label (487K+)
df_60_40_train = df_60_40.groupby("label", group_keys=False).apply(lambda x: x.sample(n=2000, random_state=42)) # 2000 from each label (27K+)
df_human_train = df_human.groupby("label", group_keys=False).apply(lambda x: x.sample(n=65, random_state=42)) # Only 85 examples of AI texts, we'll save some for test

print("--- Create df_test ---")
df_1k_test = df_1k.drop(df_1k_train.index)
df_41k_test = df_41k.drop(df_41k_train.index)
df_487k_test = df_487k.drop(df_487k_train.index)
df_60_40_test = df_60_40.drop(df_60_40_train.index)
df_human_test = df_human.drop(df_human_train.index)

df_test = pd.concat([df_1k_test, df_41k_test, df_487k_test, df_60_40_test, df_human_test], ignore_index=True)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

print("--- Create df_train ---")
df_train = pd.concat([df_1k_train, df_41k_train, df_487k_train, df_60_40_train, df_human_train], ignore_index=True)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

print("--- Save as csv files ---")
folder_path = "/home/joberant/NLP_2425b/maozhaim/datasets"
os.makedirs(folder_path, exist_ok=True)

train_path = os.path.join(folder_path, "train_dataset.csv")
test_path = os.path.join(folder_path, "test_dataset.csv")

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

print(f"Saved train_dataset in {train_path}")
print(f"Saved test_dataset in {test_path}")

print("--- END ---")
