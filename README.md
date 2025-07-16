# 📰 Fake News Detection using NLP and Deep Learning

This project is a **Fake vs Real News Classifier** built using **Natural Language Processing (NLP)** techniques and **Deep Learning**. It leverages a custom neural network architecture including Conv1D and Bidirectional GRU layers to accurately classify news articles as either **Fake** or **Real**.

---

## 📌 Table of Contents
- [📂 Dataset](#-dataset)
- [🛠️ Tools & Technologies](#-tools--technologies)
- [🚀 Project Pipeline](#-project-pipeline)
- [📊 Model Performance](#-model-performance)
- [💾 How to Run](#-how-to-run)
- [📈 Future Improvements](#-future-improvements)
- [📜 License](#-license)

---

## 📂 Dataset
- Source: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Files used:
  - `True.csv` – Real news articles
  - `Fake.csv` – Fake news articles
- Data Fields: `title`, `text`, `subject`, `date`

---

## 🛠️ Tools & Technologies
- Python 🐍
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- NLP preprocessing (Regex, Tokenizer, Padding)

---

## 🚀 Project Pipeline

1. **Data Loading**
   - Loaded both real and fake news datasets
   - Combined and labeled the data

2. **Text Preprocessing**
   - Cleaned HTML tags, URLs, punctuation, and numbers
   - Removed extra whitespace and lowercased text

3. **Tokenization & Padding**
   - Converted text to sequences using `Tokenizer`
   - Standardized input length with `pad_sequences`

4. **Model Architecture**
   - Embedding layer
   - Bidirectional GRU for sequential context
   - Dense layers with `sigmoid` activation for binary classification

5. **Training & Evaluation**
   - Trained with `binary_crossentropy` loss
   - Visualized training history
   - Evaluated using confusion matrix and classification report

6. **Model Saving**
   - Model saved in `.keras` format

---

## 📊 Model Performance
- Achieved high training & validation accuracy
- Evaluated using:
  - 📉 Loss and Accuracy Curves
  - ✅ Classification Report
  - 🔍 Confusion Matrix

---

## 💾 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook news_detection.ipynb
