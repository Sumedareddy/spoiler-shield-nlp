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
| Accuracy  | 79.3% |
| Precision | 82.6% |
| Recall    | 87.3% |
| F1-score  | 84.9% |

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

# Spoiler Shield NLP 🚫🗣️

**An NLP-based spoiler detection system using contrastive learning and transformer embeddings on Reddit comments.**

Spoiler Shield automatically identifies and classifies spoiler content in user-generated movie and television discussions. Built with Reddit data, Sentence-BERT, and contrastive learning, it can be used interactively via a Streamlit web app.

---

## 📌 Table of Contents

* [Problem Statement](#problem-statement)
* [Solution Approach](#solution-approach)
* [Features](#features)
* [Pipeline Overview](#pipeline-overview)
* [Dataset](#dataset)
* [Model Training](#model-training)
* [Evaluation Results](#evaluation-results)
* [Live Demo](#live-demo)
* [Installation](#installation)
* [Usage](#usage)
* [References](#references)

---

## 🔍 Problem Statement

Social media platforms are flooded with spoiler-laden comments that can ruin the experience for users who haven't seen a film or show yet. Spoiler Shield aims to detect these spoilers automatically by distinguishing them from general discussions using semantic similarity and machine learning.

---

## 🧠 Solution Approach

* Scrape Reddit comments from spoiler-heavy subreddits
* Preprocess and clean the data (remove spoiler tags, punctuation, and stopwords)
* Use **contrastive learning** to structure spoiler/non-spoiler representations
* Train a **Sentence-BERT model** with cosine similarity loss
* Build a **real-time prediction interface** using Streamlit

---

## 🚀 Features

* ✅ Automated Reddit data collection (spoiler & non-spoiler balanced)
* ✅ Preprocessing pipeline with NLTK
* ✅ Sentence embedding training using `distilbert-base-uncased`
* ✅ Real-time spoiler classification using cosine similarity to anchor embeddings
* ✅ Deployment-ready via Streamlit + pyngrok
* ✅ Supports interactive testing for user-entered text

---

## 🔄 Pipeline Overview

```
Reddit Scraper (asyncpraw)
        ↓
Raw Comment Dataset (.csv)
        ↓
Text Preprocessing (NLTK, Regex)
        ↓
Contrastive Pair Generation
        ↓
Sentence-BERT Training (CosineSimilarityLoss)
        ↓
Embedding Analysis & Model Evaluation
        ↓
Real-Time Prediction Web App (Streamlit)
```

---

## 📂 Dataset

Collected from Reddit using the following subreddits:

* `r/movies`
* `r/television`
* `r/marvelstudios`
* `r/MovieDetails`

Spoiler comments were identified using `>!spoiler!<` markdown syntax. Cleaned dataset includes:

| Filename                     | Description                     |
| ---------------------------- | ------------------------------- |
| `spoiler_shield_dataset.csv` | Raw labeled comments            |
| `spoiler_shield_cleaned.csv` | Preprocessed comments (cleaned) |

---

## 🏋️‍♂️ Model Training

* **Architecture:** Sentence-BERT (Siamese Transformer)
* **Base Model:** `distilbert-base-uncased`
* **Loss Function:** CosineSimilarityLoss
* **Batch Size:** 16
* **Epochs:** 3

Positive and negative pairs were used for contrastive learning:

* Positive: Same label (spoiler–spoiler or non-spoiler–non-spoiler)
* Negative: Mixed label (spoiler vs non-spoiler)

---

## 📊 Evaluation Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.865 |
| Precision | 0.872 |
| Recall    | 0.857 |
| F1-Score  | 0.864 |

Cosine similarity distributions and t-SNE embeddings confirm that the model effectively separates spoiler from non-spoiler comments.

To visualize training loss, similarity scores, and embeddings, use the provided Jupyter/Colab-ready scripts under the `graphs/` section of the repo.

---

## 🌐 Live Demo

A **Streamlit interface** allows real-time prediction:

```bash
streamlit run app.py
```

Example Output:

* **Input**: "He dies in the end"
* **Prediction**: Spoiler
* **Spoiler Similarity**: 0.68
* **Non-Spoiler Similarity**: 0.45

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/spoiler-shield-nlp.git
cd spoiler-shield-nlp

# Install dependencies
pip install -r requirements.txt

# Optional: Enable ngrok for demo deployment
pip install pyngrok
```

---

## 🧪 Usage

### 🟢 Train Model

```python
# Inside training script or notebook
model.fit(...)
model.save("spoiler-shield-contrastive-model")
```

### 🟢 Predict Spoiler

```python
from inference import predict_spoiler

comment = "This twist at the end shocked everyone!"
result = predict_spoiler(comment)
print(result)
```

### 🟢 Run Web App

```bash
streamlit run app.py
```

---

## 📚 References

* [Sentence-BERT: EMNLP 2019](https://arxiv.org/abs/1908.10084)
* [BERT for Spoiler Detection: IEEE Access](https://ieeexplore.ieee.org/document/9157670)
* [asyncpraw Documentation](https://praw.readthedocs.io/)
* [NLTK Stopwords](https://www.nltk.org/)
* [Streamlit](https://streamlit.io/)
* [Hugging Face Transformers](https://huggingface.co/transformers/)

---

## 📌 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

Special thanks to the open-source community and academic researchers whose work inspired this project.


