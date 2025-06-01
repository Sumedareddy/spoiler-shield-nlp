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
| Accuracy  | 86.0% |
| Precision | 84.0% |
| Recall    | 85.0% |
| F1-Score  | 84.5% |

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
Comment: He dies saving the others.
Spoiler Similarity Score: 0.7588
Non-Spoiler Similarity Score: 0.3611
Prediction: Spoiler
```
<br><br><br>
![image](https://github.com/user-attachments/assets/d30ae3b4-4561-47a0-ac6b-0d4d1ed780fb)


### Non-Spoiler Prediction

```
Comment: The cinematography was breathtaking.
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
