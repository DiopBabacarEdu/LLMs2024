
# Fake News Classification Using BERT

This script demonstrates how to build a fake news classification pipeline using the **BERT** model. It includes data preparation, preprocessing, visualization, and model fine-tuning.

---

## Step 1: Install Required Libraries

Install necessary libraries for accessing datasets, processing text, and training models.

```bash
!pip install -q kaggle gdown transformers
```

---

## Step 2: Setup Kaggle API for Data Download

1. Upload your `kaggle.json` API key.
2. Configure the Kaggle environment to download the fake news dataset.

```python
from google.colab import files
files.upload()  # Upload kaggle.json file

!rm -rf ~/.kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download Fake News Dataset
!kaggle competitions download -c fake-news
!unzip test.csv.zip
!unzip train.csv.zip
```

Additionally, download supplementary data:

```bash
!gdown "https://drive.google.com/uc?id=178f_VkNxccNidap-5-uffXUW475pAuPy&confirm=t"
!unzip fake-news.zip
```

---

## Step 3: Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Step 4: Load and Explore the Data

Load the training and test datasets:

```python
news_d = pd.read_csv("train.csv")
submit_test = pd.read_csv("test.csv")

print("Shape of News data:", news_d.shape)
print("Columns in dataset:", news_d.columns)

# Display first rows of the dataset
news_d.head()
```

### Text Statistics

Analyze word-level statistics:

```python
txt_length = news_d.text.str.split().str.len()
title_length = news_d.title.str.split().str.len()

print(txt_length.describe())
print(title_length.describe())

sns.countplot(x="label", data=news_d)
```

---

## Step 5: Preprocess the Data

### Cleaning Text

Clean and preprocess text:

```python
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

stop_words = stopwords.words('english')
wnl = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove special characters
    text = text.strip()
    return text

def preprocess(text):
    text = clean_text(text)
    words = [wnl.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

news_d["text"] = news_d["text"].apply(preprocess)
news_d["title"] = news_d["title"].apply(preprocess)

news_d.head()
```

---

## Step 6: Visualize the Data

### WordCloud

```python
from wordcloud import WordCloud

wordcloud = WordCloud(background_color="black", width=800, height=600)
text_cloud = wordcloud.generate(" ".join(news_d['text']))

plt.figure(figsize=(10, 8))
plt.imshow(text_cloud)
plt.axis("off")
plt.show()
```

### N-Gram Analysis

```python
from nltk import ngrams, FreqDist

def plot_top_ngrams(corpus, title, n=2):
    ngrams_freq = FreqDist(ngrams(corpus.split(), n))
    ngrams_freq.most_common(10)

    plt.figure(figsize=(10, 5))
    plt.barh([f"{' '.join(ngram)}" for ngram, freq in ngrams_freq.most_common(10)],
             [freq for ngram, freq in ngrams_freq.most_common(10)])
    plt.title(title)
    plt.show()

plot_top_ngrams(" ".join(news_d[news_d.label == 0]['text']), "Top Bigrams in Reliable News")
plot_top_ngrams(" ".join(news_d[news_d.label == 1]['text']), "Top Bigrams in Fake News")
```

---

## Step 7: Fine-tune BERT for Classification

### Import BERT

```python
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
```

### Prepare Data

Split the dataset and tokenize text:

```python
from sklearn.model_selection import train_test_split

def prepare_data(df):
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
    return train_encodings, valid_encodings, train_labels, valid_labels

train_encodings, valid_encodings, train_labels, valid_labels = prepare_data(news_d)
```

### Create Dataset Class

```python
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}, torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)
```

Convert encodings:

```python
train_dataset = NewsDataset(train_encodings, train_labels)
valid_dataset = NewsDataset(valid_encodings, valid_labels)
```

### Initialize and Train the Model

```python
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="steps",
    save_steps=500,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

trainer.train()
```

---

## Step 8: Evaluate and Predict

Save the model and evaluate on test data:

```python
model.save_pretrained("fake-news-bert")
tokenizer.save_pretrained("fake-news-bert")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

print(predict("This is a fake news article."))
```

---

## Step 9: Submission File

Prepare submission file:

```python
test_df = pd.read_csv("test.csv")
test_df["text"] = test_df["text"].apply(preprocess)
test_df["label"] = test_df["text"].apply(predict)
test_df[["id", "label"]].to_csv("submission.csv", index=False)
```
