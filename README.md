# 🛰️ Hyperspectral Image Analysis Dashboard

A **Streamlit-based Web Application** for analyzing and classifying Hyperspectral Imaging (HSI) datasets using Machine Learning techniques.

---

## 📖 Overview

This project transforms a traditional Python-based workflow into a **fully interactive web application**. It allows users to upload hyperspectral `.mat` files, explore spectral bands, apply preprocessing techniques like PCA, train ML models, and visualize classification results — all in real-time.

The application is designed for **remote sensing, land cover classification, and data science experimentation**.

---

## 🚀 Key Features

### 🛠️ Software Engineering Highlights

* ✅ Interactive UI built with **Streamlit**
* ⚡ Optimized performance using `@st.cache_data`
* 🧩 Modular architecture (separate utils for clean code)
* 💾 Model saving support (`.pkl` format)
* 🎛️ Dynamic parameter tuning (no code changes required)

---

### 📊 Data Science Capabilities

* 📈 Exploratory Data Analysis (EDA) with band visualization
* 🧠 Dimensionality Reduction using **PCA**
* 🤖 Model selection:

  * k-Nearest Neighbors (kNN)
  * Gaussian Naive Bayes
* 📉 Confusion Matrix & Classification Report
* 🎯 Accuracy calculation
* 🗺️ Ground truth visualization

---

## 🧱 Project Structure

```
hyperspectral-dashboard/
│
├── app.py
├── requirements.txt
│
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── visualization.py
│
├── models/
└── data/
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/hyperspectral-dashboard.git
cd hyperspectral-dashboard
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🏃‍♂️ Usage

Run the Streamlit app:

```
streamlit run app.py
```

---

## 📂 Dataset

This application works with hyperspectral `.mat` files.

### Recommended Dataset:

* Indian Pines Dataset

Download from:
👉 https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes

Required files:

* `Indian_pines_corrected.mat` (image data)
* `Indian_pines_gt.mat` (ground truth labels)

---

## 🧪 How It Works

1. Upload hyperspectral image and ground truth files
2. Visualize spectral bands interactively
3. (Optional) Apply PCA for dimensionality reduction
4. Select ML model and configure parameters
5. Train model and view results instantly

---

## 📸 Screenshots

*(Add screenshots here after running the app)*

---

## 🤝 Contributing

Contributions are welcome!

* Fork the repository
* Create a new branch
* Submit a pull request

---

## 🧠 Tech Stack

* Python
* Streamlit
* NumPy
* SciPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## 🎯 Future Improvements

* 🌍 Deployment with cloud dataset support
* 🗺️ Predicted map visualization
* 📥 Download predictions
* ⚡ Deep learning integration (CNNs)

---

## 👩‍💻 Author

**Kinza Zahra**

---

## ⭐ Show Your Support

If you like this project, please ⭐ the repository!

---
