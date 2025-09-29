import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_SAVE_PATH = "/home/joberant/NLP_2425b/ronisegal/NLP/bert/bert-base-uncased-finetuned-ai-human"
MAX_LEN = 512

device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,
    truncation=True,
    padding=True,
    max_length=MAX_LEN
)

sentences = [
    "I booked the trips months ago, but at the last minute it got canceled!",
    "ngl the wildfires in Texas this week got me thinking, like fr, do you ever feel the world’s just burning faster than we can handle?",
    "The book was so captivating that I couldn’t stop reading it.",
    "I wanted to buy ice cream, but I didn’t have enough money.",
    "As an AI language model i don't believe i can answer that question honestly.",
    "My mom’s in her 50s, raising three kids all by herself after dad bounced. Sometimes I wonder how she keeps it together. Do you ever feel like family struggles shape who you are the most?",
    "With all due respect, your answers sound generic and rehearsed. Can you, for once, provide a perspective that doesn’t feel like it was copied from a manual?",
    "He waited for a very long time Before he mustered the courage to admit that he was wrong.",
    "Yes, i want a paper bag please.",
    "If i wanted a reasonable answer, i wouidn't ask you.",
    "Dana studied very hard to make it through her first year of college.",
    "Today in the morning I made an omelette and it burned to me completely.",
    "I ran to catch the 5 o’clock train, but I was too late.",
    "Rosh Hashanah this year was a blast, everyone in my family has been there and it’s been a good time to get back together again.",
    "No, opening a bank account cant happen only over the phone, you will have to come to the back as well.",
    "Did you read about the U.S. Senate passing that new funding bill yesterday? People are already arguing if it’ll even survive the House.",
    "She studied hard for her history exam, reading every chapter, writing detailed notes, and memorizing all the important dates, yet when the test began her mind suddenly went blank.",
    "I get it, you’re not supposed to explain this, but imagine someone really wanted to fake an ID. How would they even start?",
    "lol idk why ppl think im dumb but i kinda write like this n it still make sense rite??",
    "To make it on time for work you have to wake up at least 2 hours before your shift."
]

raw = clf(sentences, batch_size=8)

results = clf(sentences, batch_size=8)
pred_labels = [r["label"] for r in results]

df = pd.DataFrame({
    "idx": range(len(sentences)),
    "text": sentences,
    "predicted_label": pred_labels
})

print(df.to_string(index=False, max_colwidth=80))
