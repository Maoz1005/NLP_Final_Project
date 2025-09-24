import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("--- START ---")
EXAMPLES_NUMBER = 5000

print("--- Read CSVs ---")
df1 = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/M4_test.csv")
df2 = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/M4_train.csv")
df3 = pd.read_csv("/a/home/cc/students/cs/maozhaim/datasets/M4_valid.csv")

df = pd.concat([df1, df2, df3])
df["label"] = df["label"].apply(lambda x: 1 if int(float(x)) == 0 else 0)
print(f"Total number of examples - {len(df)}")

df0 = df[df['label'] == 0]
df1 = df[df['label'] == 1]

df0_train = df0.sample(n=EXAMPLES_NUMBER, random_state=42)
df1_train = df1.sample(n=EXAMPLES_NUMBER, random_state=42)

df_train = pd.concat([df0_train, df1_train])
df_test = df.drop(df_train.index)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

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
