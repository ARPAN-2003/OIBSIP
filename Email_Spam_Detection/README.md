# 📧 Premium Email Spam Detector

<div align="center">

### 🚀 AI Powered Email & SMS Spam Detection using Machine Learning and NLP

🌐 **Live Demo:** https://email-spam-detector-isrx.onrender.com

Detect whether an Email or SMS is **Spam** or **Genuine (Ham)** using
**Natural Language Processing (NLP)**, **TF-IDF Vectorization**, and
**Multinomial Naive Bayes Classification**.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge)

</div>

---

# 📌 Project Overview

Spam emails and SMS messages are becoming increasingly common.

This project uses **Machine Learning** and **Natural Language Processing (NLP)** techniques to classify messages into:

✅ Ham (Genuine)

🚨 Spam (Suspicious)

The application also provides:

- Prediction Confidence
- Spam/Ham Probability
- Dataset Analytics
- Word Clouds
- Interactive Dashboard

---

# ✨ Features

- 📧 Email & SMS Spam Detection
- 🤖 Machine Learning Classification
- 🧠 NLP Text Preprocessing
- 📊 Prediction Confidence Score
- 📈 Spam vs Ham Analytics
- ☁️ Word Cloud Visualization
- 📏 Message Length Analysis
- 🎨 Modern Streamlit Dashboard
- ⚡ Real-Time Prediction

---

# 🖥️ Demo

## Home Dashboard

```
Premium Email Spam Detector

AI Powered NLP + Machine Learning + TF-IDF + Naive Bayes
```

### Prediction Result

```
🚨 SPAM MESSAGE DETECTED

Risk Level : HIGH

Confidence : 98.73%
```

---

# 📂 Project Structure

```
Email_Spam_Detection/

│

├── data/

│   └── spam.csv

│

├── models/

│   ├── spam_model.pkl

│   └── vectorizer.pkl

│

├── outputs/

│   ├── accuracy_graph.png

│   └── confusion_matrix.png

│

├── src/

│   ├── __init__.py

│   ├── preprocess.py

│   ├── train.py

│   ├── predict.py

│   └── visualize.py

│

├── app.py

├── main.py

├── requirements.txt

└── README.md
```

---

# 🧠 Machine Learning Pipeline

```
Raw Email / SMS

↓

Text Cleaning

↓

Lowercase Conversion

↓

Tokenization

↓

Stopword Removal

↓

Stemming

↓

TF-IDF Vectorization

↓

Multinomial Naive Bayes

↓

Spam / Ham Prediction
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|-----------------|-----------------------------|
| Python | Programming Language |
| Pandas | Data Processing |
| NLTK | Text Preprocessing |
| TF-IDF | Feature Extraction |
| Scikit-Learn | Machine Learning |
| Multinomial Naive Bayes | Classification |
| Streamlit | Web Application |
| Matplotlib | Data Visualization |
| WordCloud | Text Visualization |

---

# 📊 Dataset Information

Dataset Used:

**SMS Spam Collection Dataset**

Total Messages

```
5572
```

Ham Messages

```
4825
```

Spam Messages

```
747
```

---

# 📈 Model Performance

| Metric | Score |
|----------------|------------|
| Accuracy | **97.58%** |
| Algorithm | Multinomial Naive Bayes |
| Vectorizer | TF-IDF |
| Classes | Spam / Ham |

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/ARPAN-2003/OIBSIP.git
```

Move into project

```bash
cd OIBSIP/Email_Spam_Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

### Train Model

```bash
python src/train.py
```

### Run CLI Version

```bash
python main.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

# 📸 Outputs

Generated automatically:

```
outputs/

accuracy_graph.png

confusion_matrix.png
```

---

# 🔮 Future Improvements

- 📂 Drag & Drop Email File Upload
- 📄 PDF Report Generation
- ☁️ Cloud Deployment
- 📧 Bulk Email Detection
- 🤖 Deep Learning Models
- 🌐 API Integration

---

# 👨‍💻 Author

**Arpan Mitra**

AICTE Oasis Infobyte Internship Project

Machine Learning • NLP • Python • Streamlit

---

# ⭐ If you like this project

Give it a ⭐ on GitHub and share it with others!
