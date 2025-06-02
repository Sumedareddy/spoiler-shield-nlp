# Spoiler Shield with NLP 

**Spoiler Shield** is a contrastive learning-based NLP project designed to detect spoilers in user-generated movie and TV show comments. Leveraging Reddit data and state-of-the-art transformer models, this system classifies comments as **spoilers** or **non-spoilers**, even when the spoilers are subtly or indirectly expressed. It offers a real-time interactive prediction app powered by **Streamlit**.

---

## Problem Overview

In an era where movie discussions are prevalent across social platforms, spoilers often appear—accidentally or deliberately—within user comments. Traditional systems use keyword matching, which fails to detect rephrased or implicit spoilers. This project introduces a semantic-based method using contrastive learning and Sentence-BERT embeddings.

---

## Objectives

* Build a high-quality spoiler detection model using **contrastive learning**.
* Use **SentenceTransformer (DistilBERT-based)** models for deep text representation.
* Create a balanced dataset of Reddit movie comments.
* Train on **positive (same class)** and **negative (opposite class)** comment pairs.
* Deploy an interactive **Streamlit** app for real-time prediction.

---

## Project Architecture

* **Language**: Python
* **Libraries**: `asyncpraw`, `pandas`, `nltk`, `sentence-transformers`, `torch`, `streamlit`
* **Model**: Contrastive learning with `CosineSimilarityLoss` and `distilbert-base-uncased`
* **Training Data**: 2000 Reddit comments (1000 spoilers, 1000 non-spoilers)
* **Deployment**: Web UI using Streamlit and PyNgrok


## Project Files

| File                                                | Description                                        |
| --------------------------------------------------- | -------------------------------------------------- |
| `spoiler_shield_dataset.csv`                        | Raw Reddit comments with labels                    |
| `spoiler_shield_cleaned.csv`                        | Cleaned and normalized comment text                |
| `spoiler-shield-contrastive-model/`                 | Trained Sentence-BERT model                        |
| `SpoilerShield_using_NLP&ContrastiveLearning.ipynb` | Complete training and inference pipeline           |
| `app.py`                                            | Streamlit app script                               |
| `Project_Final_Report.pdf`                          | Full project documentation with graphs and results |


---

## Implementation

This section outlines the end-to-end pipeline built for spoiler detection using contrastive learning and transformer-based NLP methods.

Step 1: Data Collection (Reddit Scraper)
* Reddit comments are collected using asyncpraw from subreddits: r/movies, r/television, r/marvelstudios, r/MovieDetails
* Comments are filtered based on the presence of "spoiler" keyword and Reddit’s spoiler tag syntax >!spoiler!<.
* A balanced dataset of 1000 spoiler and 1000 non-spoiler comments is saved in:
    * data/spoiler_shield_dataset.csv
 <br>
 Step 2: Text Preprocessing
* Performed using nltk and re:
    * Markdown tags removed (>!spoiler!<)
    * Lowercasing, punctuation and stopword removal
* Resulting dataset saved in:
    * data/spoiler_shield_cleaned.csv 
Cleaned comments are used for both training and embedding generation.
<br>
Step 3: Contrastive Learning Dataset Preparation
* From the cleaned data, comment pairs are generated as:
    * Positive pairs: both spoiler or both non-spoiler
    * Negative pairs: one spoiler, one non-spoiler
* Up to 2000 training pairs are created using sentence-transformers.InputExample
<br>
Step 4: Model Training (Sentence-BERT)
* Pretrained transformer: distilbert-base-uncased
* Contrastive learning using CosineSimilarityLoss
* Training pipeline:
    * Batched in DataLoader
    * 3 epochs
    * Model saved to:
        * model/spoiler-shield-contrastive-model/
<br>
Step 5: Semantic Embedding & Anchor Generation
* Trained model is used to embed all spoiler and non-spoiler comments.
* Mean vector for each class is computed as its anchor.
* Anchors saved using PyTorch for real-time use.
<br>
Step 6: Evaluation
* A test set of 300 spoiler and 300 non-spoiler samples was used to evaluate the trained model.
* Cosine similarity between embedded comment pairs was used to classify them as similar (same class) or dissimilar (opposite class).
* The model was evaluated using standard classification metrics:

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 73.7% |
| Precision | 77.2% |
| Recall    | 86.0% |
| F1-Score  | 81.3% |


## Key Visualizations (from report)

![image](https://github.com/user-attachments/assets/e80d059d-3b36-47d5-9960-6707df6aa63f)

Figure 1: Displays the count of spoiler vs non-spoiler comments, confirming dataset class balance.<br><br><br>



![image](https://github.com/user-attachments/assets/fc6faaf4-fd74-4621-8039-ba5fc593daef)

Figure 2: Shows a reduction in average word count after preprocessing, improving signal-to-noise ratio.<br><br><br>



![image](https://github.com/user-attachments/assets/5319416e-abcc-4bfc-a63e-b6c570a1a35b)

Figure 3: Demonstrates that cosine similarity scores between embeddings form clearly separable clusters.<br><br><br>

---

## Model Performance

![image](https://github.com/user-attachments/assets/a3b137c9-adc0-440a-9525-7f9c3bca57e2)

Figure 4: Compares final performance metrics (Accuracy, Precision, Recall, F1); all exceed 85% threshold.<br><br><br>


| Metric    | Value |
| --------- | ----- |
| Accuracy  | 73.7% |
| Precision | 77.2% |
| Recall    | 86.0% |
| F1-Score  | 81.3% |

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install asyncpraw pandas nltk sentence-transformers torch streamlit pyngrok
```

### 2. Train Model (Optional)

Use the `.ipynb` notebook to:

* Scrape Reddit
* Preprocess comments
* Generate contrastive pairs
* Train the model with cosine similarity loss

The trained model is saved in `model/spoiler-shield-contrastive-model`.

### 3. Launch Web App

```bash
streamlit run app.py
```

For public demo using `pyngrok`:

```python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("App is live at:", public_url)
```

### 4. Try Predictions

Enter any comment and receive a prediction:

* **Spoiler**
* **Non-Spoiler**

Scores for both classes are shown.

---

## Sample Outputs
![image](https://github.com/user-attachments/assets/e646c0e8-f192-4f10-b58f-6cf1bc374569)

### Spoiler Prediction

```
Comment: I can't believe they killed off the main character in the last episode! That scene where Jon stabs Daenerys was brutal.
Spoiler Similarity Score: 0.7588
Non-Spoiler Similarity Score: 0.3611
Prediction: Spoiler
```
<br><br><br>
![image](https://github.com/user-attachments/assets/d30ae3b4-4561-47a0-ac6b-0d4d1ed780fb)


### Non-Spoiler Prediction

```
Comment: I like the movie director.
Spoiler Similarity Score: 0.3807
Non-Spoiler Similarity Score: 0.6972
Prediction: Non-Spoiler
```

---

## References

1. Allen Bao, et al., “Spoiler Alert: Using NLP to Detect Spoilers in Book Reviews”, 2021.
2. Golbeck, J. (2015), “Detecting Spoilers in Fan Wikis”, HICSS.
3. Boyd-Graber, J., et al. (2019), “Spoiler Alert: Machine Learning Approaches”, NAACL.
4. Chang & Huang (2020), “BERT for Spoiler Detection”, IEEE Access.
5. Reimers & Gurevych (2019), “Sentence-BERT: Siamese Networks”, EMNLP.
6. [PRAW Reddit API](https://praw.readthedocs.io/en/latest/)
7. [SentenceTransformers](https://www.sbert.net/)
8. [Hugging Face Transformers](https://huggingface.co/)
9. [NLTK](https://www.nltk.org/)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

Special thanks to **Dr. Vivek Kumar Mishra** (Supervisor) and Mahindra University for supporting this work as part of the M.Tech dissertation.
