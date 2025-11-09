# 🧠 Basic NLP on Disaster Tweets — Technical Report

## 📘 Project Overview
This project applies **Natural Language Processing (NLP)** techniques to classify tweets as **disaster-related or not**.  
The dataset comes from the [Kaggle “Real or Not? Disaster Tweets” competition](https://www.kaggle.com/c/nlp-getting-started).  
The objective is to build a model that can automatically detect whether a tweet refers to a real disaster event.

---

## 🧩 Dataset Description
- **Source:** Kaggle Dataset (`train.csv`, `test.csv`)
- **Columns:**
  - `id` — Unique identifier of the tweet  
  - `text` — Tweet content  
  - `keyword` — Disaster-related keyword (may contain nulls)  
  - `location` — User location (optional)  
  - `target` — 1 if the tweet describes a real disaster, 0 otherwise  

Dataset includes **7,613 labeled tweets** for training and **3,263** for testing.

---

## 🔍 Exploratory Data Analysis (EDA)
- Detected and removed **52 duplicate rows**.  
- Verified class distribution: approximately **57% non-disaster**, **43% disaster** tweets.  
- Common disaster-related keywords include *fire*, *flood*, *earthquake*, and *storm*.  
- Used **WordClouds** and **bar charts** to visualize top words per class.

---

## 🧼 Text Preprocessing
Pipeline steps for tweet cleaning:
1. Convert text to lowercase  
2. Remove URLs, punctuation, and special symbols  
3. Tokenize using **NLTK** or **spaCy**  
4. Remove stopwords  
5. Lemmatize tokens  
6. Handle missing values in `keyword` and `location`  

These steps ensure consistent input for feature extraction.

---

## ⚙️ Feature Engineering
Explored multiple text representations:
- **Bag-of-Words (CountVectorizer)**  
- **TF-IDF Vectorizer** (Term Frequency–Inverse Document Frequency)

Additional features tested:
- Encoded `keyword` using one-hot encoding  
- Added simple numerical features like tweet length and punctuation counts  

---

## 🤖 Model Training and Comparison
| Model | Vectorizer | Notes |
|--------|-------------|-------|
| Logistic Regression | TF-IDF | Strong baseline, interpretable, fast convergence |
| Multinomial Naïve Bayes | TF-IDF | Performs well on sparse data |
| Random Forest | CountVectorizer | Lower accuracy, slower |
| LSTM / SimpleRNN | Tokenized sequences | Captures semantics but needs more data |

**Best model:** Logistic Regression with TF-IDF → **≈80–82% validation accuracy**

---

## 📊 Evaluation Metrics
Metrics used:
- **Accuracy**
- **Precision / Recall / F1-score**
- **Confusion Matrix**
- **k-Fold Cross Validation**

Example results:
> F1-score = 0.81, Precision = 0.80, Recall = 0.82

---

## 🚀 Inference and Kaggle Submission
- Preprocess test set using the same pipeline  
- Generate predictions from best model  
- Save output to `submission.csv` in format:
  ```csv
  id,target
  1,0
  2,1
  ...
