# 🔐 Machine Learning-Based Framework for Early Detection of Data Leakage in Multi-Tenant Cloud Environments

## 📌 Overview

This project presents an intelligent and secure framework for detecting **data leakage in multi-tenant cloud environments** using Machine Learning techniques.

The system integrates:

* Predictive analytics
* Data encryption (RSA)
* Cloud storage security
* Real-time alert mechanisms

It helps identify whether data is **“Leaked” or “Not Leaked”** and ensures proactive monitoring and protection of sensitive information.

---

## 🚀 Features

* 🔍 **Data Leakage Detection**

  * Uses Machine Learning models to detect suspicious patterns
* 📊 **Model Comparison**

  * Random Forest vs 1D CNN
* 🔐 **RSA Encryption**

  * Secures sensitive data before storage
* 📩 **Email Alerts**

  * Sends real-time alerts on leakage detection
* 📈 **Data Visualization**

  * Confusion Matrix, ROC Curve, PCA plots
* 👨‍💻 **Admin Dashboard**

  * Monitor users, datasets, and leakage logs
* 📁 **Report Generation**

  * Export detailed analysis in PDF format

---

## 🧠 Technologies Used

* **Programming Language:** Python
* **Framework:** Flask
* **Libraries:**

  * Pandas, NumPy
  * Scikit-learn
  * TensorFlow / Keras
  * Matplotlib, Seaborn
* **Security:** RSA Encryption
* **Frontend:** HTML, CSS
* **Cloud Integration:** Cloud-based storage (simulated/CloudMe)

---

## ⚙️ System Architecture

The system follows a structured ML pipeline:

1. Data Collection
2. Data Preprocessing
3. Feature Extraction (PCA)
4. Model Training
5. Prediction
6. Encryption
7. Alert System

---

## 🏗️ Project Workflow

```
User Upload → Preprocessing → PCA → Model Training
→ Prediction → Encryption → Cloud Storage → Email Alert
```

---

## 📂 Dataset

* Input formats supported:

  * CSV
  * XLSX
* Includes multi-tenant cloud activity data such as:

  * User access
  * Resource usage
  * Data transfer
  * Authentication methods

---

## 🤖 Machine Learning Models

### 1. Random Forest Classifier

* Handles structured data efficiently
* High accuracy and robustness

### 2. 1D Convolutional Neural Network (CNN)

* Captures complex patterns
* Useful for sequential/behavioral data

👉 Best model is automatically selected based on performance.

---

## 🔐 Security Implementation

* RSA Encryption is used to:

  * Encrypt prediction results
  * Secure stored data
* Decryption is done using private keys

---

## 📩 Email Notification System

* Triggered when data leakage is detected
* Sends:

  * Alert message
  * Encrypted result
* Uses SMTP protocol (Gmail)

---

## 💻 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/data-leakage-detection.git
cd data-leakage-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
python app.py
```

---

## 📊 Output

* Prediction:

  * **Data Leakage Detected** / **No Data Leakage**
* Graphs:

  * ROC Curve
  * PCA Visualization
* Encrypted Output File
* Email Notification

---

## 📌 Applications

* ☁️ Cloud Service Providers
* 🏥 Healthcare Systems
* 💰 Banking & Finance
* 🛒 E-commerce Platforms
* 🏛️ Government Systems
* 🎓 Educational Institutions

---

## 🔮 Future Scope

* Integration of **LSTM & Transformer models**
* **Real-time data streaming support**
* **NLP for sensitive data detection**
* **Blockchain for secure logging**
* Multi-cloud deployment

---



