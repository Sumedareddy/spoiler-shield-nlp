# Spoiler Shield NLP
A prototype app for spoiler detection using NLP and contrastive learning
**Detect spoilers in text comments using contrastive learning and sentence embeddings.**

---

## Overview

Spoiler Shield NLP is a Python-based project that detects spoilers in user-generated comments from Reddit. It leverages **contrastive learning** on sentence embeddings to distinguish spoiler comments from non-spoiler ones with high accuracy.

---

## Features

* **Reddit Data Scraping:** Automatically collects and labels comments containing spoilers from multiple subreddits such as r/movies, r/television, r/marvelstudios, and r/MovieDetails.
* **Data Preprocessing:** Cleans and normalizes text by removing spoiler markdown, URLs, special characters, and stopwords.
* **Contrastive Learning:** Trains a Sentence-BERT model to embed spoiler and non-spoiler comments close together in vector space while pushing different types apart.
* **Real-Time Spoiler Detection:** Efficient function to predict spoiler probability for any new text comment.
* **Evaluation:** Metrics including accuracy, precision, recall, and F1-score on a balanced test set.

---

## Dataset

* **Source:** Reddit comments collected via the asyncpraw API.
* **Subreddits:** `r/movies`, `r/television`, `r/marvelstudios`, `r/MovieDetails`.
* **Filtering:** Posts containing the keyword "spoiler" with comments marked by spoiler markdown (`>!spoiler!<`).
* **Size:** 1000 labeled spoiler comments and 1000 labeled non-spoiler comments.
* **Format:** CSV file with columns:

  * `Movie`: Title of the Reddit submission.
  * `Comment`: Raw comment text.
  * `Comment Type`: Label ("Spoiler" or "Non-Spoiler").
  * `Cleaned Comment`: Preprocessed comment text used for training.

---

## Model Architecture

* **Base Model:** `distilbert-base-uncased` transformer as the embedding backbone.
* **Pooling Layer:** Mean pooling to generate fixed-size sentence embeddings.
* **Training Loss:** `CosineSimilarityLoss` used for contrastive learning.
* **Training Data:** Pairs of comments — positive pairs (same label) and negative pairs (different labels).
* **Output:** A Sentence-BERT model that maps comments into an embedding space where semantic similarity correlates with spoiler classification.

---

## Installation

```bash
git clone https://github.com/Sumedareddy/spoiler-shield-nlp.git
cd spoiler-shield-nlp
pip install -r requirements.txt
```

`requirements.txt` should include:

```
asyncpraw
nest_asyncio
pandas
nltk
sentence-transformers
torch
scikit-learn
```

---

## Usage

### Step 1: Data Collection

Scrape Reddit comments containing spoilers and save them into a structured CSV.

```python
# Run the provided async scraper with your Reddit credentials
await fetch_comments()
```

### Step 2: Data Preprocessing

Clean and normalize the scraped comments.

```python
df = pd.read_csv("spoiler_shield_dataset.csv")
df["Cleaned Comment"] = df["Comment"].apply(clean_comment)
df.to_csv("spoiler_shield_cleaned.csv", index=False)
```

### Step 3: Generate Contrastive Pairs

Create pairs of text comments for contrastive training.

```python
train_examples = generate_contrastive_pairs(df)
```

### Step 4: Train the Model

Train the Sentence-BERT model with contrastive loss.

```python
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True
)
model.save("spoiler-shield-contrastive-model")
```

### Step 5: Evaluate Model

Test the model performance on a balanced test set.

```python
# Compute accuracy, precision, recall, F1-score
```

### Step 6: Real-Time Spoiler Detection

Load the trained model and classify new comments.

```python
label, score = predict_spoiler(new_comment_text)
print(f"Prediction: {label} (confidence {score:.2f})")
```

---

## Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | \~XX% |
| Precision | \~XX% |
| Recall    | \~XX% |
| F1-score  | \~XX% |

(*Replace XX with your actual evaluation results*)

---

## Future Work

* Expand dataset size and diversity.
* Fine-tune threshold for spoiler classification.
* Add multi-language support.
* Deploy model as an API or browser extension.

---

## License

MIT License © 2025 Sumeda Reddy

---

If you want, I can generate a `requirements.txt` file or a quick-start notebook too. Would you like that?
