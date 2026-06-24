# 🌸 Iris Flower Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

### AICTE Oasis Infobyte Data Science Internship (OIBSIP)

</div>

---

# 📌 Project Overview

The **Iris Flower Classification** project is a Machine Learning application that predicts the species of an Iris flower based on its measurements.

The model classifies flowers into three categories:

- 🌼 Iris-setosa
- 🌷 Iris-versicolor
- 🌹 Iris-virginica

using **Sepal Length, Sepal Width, Petal Length, and Petal Width**.

---

# 🎯 Objective

Develop a Machine Learning model capable of accurately classifying Iris flowers and provide an interactive web interface for real-time prediction.

---

# ✨ Features

✅ Interactive Streamlit Web Application

✅ Real-time Flower Species Prediction

✅ Prediction Confidence Table

✅ Top 10 Similar Matching Records

✅ Species Information Display

✅ Correlation Matrix

✅ Confusion Matrix

✅ Classification Report

✅ Feature Visualization

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|------------------------------|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Scikit-Learn | Machine Learning |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Streamlit | Web Application |
| Joblib | Model Serialization |

---

# 📂 Dataset

**Dataset:** Iris Flower Dataset

**Features**

- Sepal Length (Cm)
- Sepal Width (Cm)
- Petal Length (Cm)
- Petal Width (Cm)

**Target**

- Iris-setosa
- Iris-versicolor
- Iris-virginica

---

# 🤖 Machine Learning Model

**Algorithm Used**

✔ Random Forest Classifier

The model is trained using the Iris dataset and saved as:

```
models/iris_model.pkl
```

---

# 📊 Model Performance

| Metric | Result |
|------------|------------|
| Accuracy | 100% |
| Classification Report | ✔ |
| Confusion Matrix | ✔ |
| Correlation Matrix | ✔ |

---

# 📁 Project Structure

```
IRIS_FLOWER_CLASSIFICATION
│
├── data/
│   └── Iris.csv
│
├── models/
│   └── iris_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── heatmap.png
│   ├── pairplot.png
│   └── species_distribution.png
│
├── src/
│   ├── predict.py
│   ├── train.py
│   ├── utils.py
│   └── visualization.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/ARPAN-2003/OIBSIP.git
```

Navigate to the project folder

```bash
cd OIBSIP/IRIS_FLOWER_CLASSIFICATION
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

### Train the Model

```bash
python main.py
```

### Launch Streamlit Application

```bash
streamlit run app.py
```

---

# 📷 Application Features

- 🌸 Predict Iris Flower Species
- 📊 Prediction Confidence
- 📈 Feature Overview
- 📏 Selected Measurements
- 🔍 Top 10 Similar Records
- ℹ Species Information

---

# 🎓 Internship

This project was developed as part of the

**AICTE Oasis Infobyte Data Science Internship Program (OIBSIP).**

---

<div align="center">

### ⭐ If you like this project, consider giving it a Star!

Made with ❤️ by **Arpan Mitra**

</div>