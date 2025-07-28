import pandas as pd
import re
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel

# ------------------ LOAD DATASET ------------------
df = pd.read_csv("1429_1.csv")
df = df[['reviews.text', 'reviews.rating']].dropna()

# Binary sentiment: rating >= 4 = positive (1), else negative (0)
df['sentiment'] = df['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)

# Use a small subset
df = df.sample(n=1000, random_state=42).reset_index(drop=True)

# ------------------ CLEAN TEXT ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['cleaned'] = df['reviews.text'].apply(clean_text)

# ------------------ VADER ------------------
analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    return 1 if score >= 0.05 else 0

df['vader_pred'] = df['cleaned'].apply(vader_sentiment)
vader_acc = accuracy_score(df['sentiment'], df['vader_pred'])
print(f"✅ VADER Accuracy: {vader_acc:.4f}")

# ------------------ BERT EMBEDDINGS ------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_cls_embeddings(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.extend(cls)
    return np.array(embeddings)

print("Extracting BERT embeddings...")
X = get_cls_embeddings(df['cleaned'].tolist())
y = df['sentiment'].values

# ------------------ Logistic Regression ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
bert_preds = clf.predict(X_test)

bert_acc = accuracy_score(y_test, bert_preds)
print(f"✅ BERT + Logistic Regression Accuracy: {bert_acc:.4f}")