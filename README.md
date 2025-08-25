# 🐦 Tweet Sentiment App

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Sentiment140-blue)](https://www.kaggle.com/datasets/kazanova/sentiment140)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow)](https://www.python.org/)

## 📌 Overview
This project implements **sentiment analysis on 1.6M tweets** using the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).  

It demonstrates the **end-to-end ML lifecycle**:
- **Preprocessing** noisy tweets (URLs, mentions, hashtags)
- **Training** a scalable pipeline (HashingVectorizer + TF-IDF + SGDClassifier)
- **Evaluation** with confusion matrix, ROC/PR curves, and error analysis
- **Deployment** as an interactive **Streamlit app** (single + batch predictions)

⚡ A strong ** project** showcasing applied NLP, scalable ML training, and real-world deployment.

---



## 🧠 Dataset
**Sentiment140**: 1,600,000 English tweets labeled for **polarity**.  
Public release contains **two classes**: `0 = negative`, `4 = positive` (no neutral rows).  
We map to string labels: **negative / positive**.

> Source: Kaggle – “Sentiment140 dataset with 1.6 million tweets”.

---

## 🚀 Quickstart (Google Colab)

> This repo assumes you’ll **train** and **serve** the app in Colab, then push artifacts + notebook to GitHub for your portfolio/demo.

### 1) Training & Evaluation (Notebook)
Open `notebooks/Sentiment140_Analysis.ipynb` in **Colab** and run cells in order.  
You’ll:
- Download the dataset via `kagglehub`
- Clean tweets and create 80/10/10 splits
- Train the scalable pipeline: `HashingVectorizer → TF-IDF → SGDClassifier (logistic)`
- Evaluate on validation & test (Macro-F1 ≈ **0.77** expected)
- Generate visuals (confusion matrix, ROC/PR, confidence histograms)
- Save the full pipeline to `models/sentiment140_sgd_hashing.pkl`


### 2) Running the Streamlit App in Google Colab (via ngrok)

You can run the **Streamlit app** directly in **Google Colab** and expose it to the web using **ngrok**.


## 🧪 Model

### Approach
The sentiment classifier is built as a **scikit-learn pipeline** with:
- **HashingVectorizer** for scalable feature extraction (fixed memory footprint, no vocab dictionary needed)  
- **TF-IDF Transformer** to emphasize informative n-grams  
- **SGDClassifier (logistic regression)** for fast, probabilistic classification with support for online learning  

### Why This Works Well
- ⚡ **Efficient & Scalable** → handles 1.6M+ tweets with fixed memory usage  
- 📈 **Strong Baseline** → TF-IDF boosts predictive power of frequent terms  
- 🔥 **Probabilities** → logistic loss provides interpretable class probabilities  

### 📊 Results (on Test Set)
- **Macro-F1 Score**: ~**0.77** (expected)  
- Balanced **precision and recall** across positive and negative classes  
- Solid baseline suitable for **real-time tweet sentiment classification**



 ## 📈 Suggested Visuals

To make evaluation and error analysis more insightful, the following visuals are recommended:

- **Confusion Matrix** → shows class-wise accuracy  
- **ROC Curve & Precision-Recall Curve** → assess classifier performance across thresholds  
- **Confidence Histograms** → distribution of predicted probabilities by true class  
- **Top Correct Predictions** → examples where the model is very confident and correct  
- **Most Confident Mistakes** → examples where the model is very confident but wrong

## 🧱 App Features

The Streamlit app provides both **single prediction** and **batch prediction** modes:

- ✅ **Single Prediction** → enter one tweet, get predicted sentiment with class probabilities (visualized as a bar chart)  
- ✅ **Batch Mode** → paste multiple tweets → get predictions in a table → download results as CSV  
- ✅ **Consistent Preprocessing** → same cleaning logic as training (removes URLs, mentions, hashtags)  
- ✅ **Interactive & Shareable** → easily demoed via ngrok tunnel in Google Colab  
 


## 🏁 Final Summary

This project demonstrates the **full machine learning lifecycle** for tweet sentiment analysis:  
- Leveraging the large-scale **Sentiment140 dataset (1.6M tweets)**  
- Building a **scalable, memory-efficient pipeline** (HashingVectorizer + TF-IDF + SGD)  
- Achieving a strong **Macro-F1 ≈ 0.77** on the test set  
- Providing **clear evaluation visuals** for interpretability  
- Deploying as a **Streamlit web app** with single and batch predictions, shareable via **ngrok**  

⚡ This makes a **practical portfolio project** that highlights skills in **NLP, scalable ML, and real-world deployment**.  


