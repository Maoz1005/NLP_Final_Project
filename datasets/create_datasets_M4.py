import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("--- START ---")

print("--- Read CSVs ---")
df1 = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/M4_test.csv")
df2 = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/M4_train.csv")
df3 = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/M4_valid.csv")

df = pd.concat([df1, df2, df3])
df['label'] = df['label'].apply(lambda x: int(float(x)))

df_test, df_train =  train_test_split(df, test_size=10000, stratify=df['label'], random_state=42)

print(f"Test size - {len(df_test)}")
print(f"Train size - {len(df_train)}")

print("--- Save as csv files ---")
folder_path = "/home/joberant/NLP_2425b/maozhaim/datasets"
os.makedirs(folder_path, exist_ok=True)

train_path = os.path.join(folder_path, "M4_train_dataset.csv")
test_path = os.path.join(folder_path, "M4_test_dataset.csv")

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

print(f"Saved train_dataset in {train_path}")
print(f"Saved test_dataset in {test_path}")

print("--- END ---")
