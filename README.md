# **📧 Spam Email Detection – Naïve Bayes vs. Logistic Regression vs. Random Forest**

## **📌 Overview**
This project implements a **spam email detection system** using machine learning models: **Naïve Bayes, Logistic Regression, and Random Forest**. The dataset used is the **Enron email dataset**, and the models are evaluated based on **accuracy, precision, and recall** to determine the best-performing classifier.

## **📂 Dataset**
The dataset consists of two columns:
- **text** → Contains the email content.
- **spam** → Binary label where **1 = Spam** and **0 = Not Spam**.

## **🔍 Data Preprocessing & Exploration**
1. **Text Cleaning & Processing**
   - Removal of special characters, numbers, and punctuation.
   - Conversion to lowercase.
   - Tokenization and stopword removal.
   - Lemmatization to normalize words.

2. **Feature Extraction Using TF-IDF**
   - **TF-IDF Vectorization** (Top 20 most important words visualized using Seaborn).

3. **Data Splitting**
   - Split into training (80%) and testing (20%) sets.

## **📊 Model Implementation & Evaluation**
Three machine learning models were implemented:
1. **Naïve Bayes Classifier**
2. **Logistic Regression**
3. **Random Forest Classifier**

Each model was evaluated based on **accuracy, precision, and recall**:

| Model                  | Accuracy | Precision | Recall  |
|------------------------|----------|------------|---------|
| Naïve Bayes           | 74.71%    | 100.00%   | 2.70%   |
| Logistic Regression   | 88.41%    | 80.60%    | 72.97%  |
| Random Forest        | 92.98%    | 87.24%    | 85.47%  |

## **📢 Conclusion**
- **Random Forest** achieved the **best performance** with **92.98% accuracy** and a good balance of **precision (87.24%)** and **recall (85.47%)**.
- **Logistic Regression** performed well and is a good choice for interpretability and efficiency.
- **Naïve Bayes**, while computationally efficient, had **low recall**, meaning it struggled to correctly identify spam emails.

## **🚀 Future Improvements**
To enhance spam detection, we can explore:
1. **Advanced Models:**
   - **SVM (Support Vector Machine)** for better decision boundaries.
   - **XGBoost** for improved classification.
   - **LSTM & Transformers (BERT, RoBERTa)** for deep learning-based email classification.

2. **Feature Engineering Enhancements:**
   - **Using Word Embeddings (Word2Vec, FastText, GloVe, or BERT)** instead of TF-IDF.
   
3. **Hyperparameter Tuning:**
   - Optimizing models using **GridSearchCV** or **RandomizedSearchCV**.
   
4. **Handling Imbalanced Data:**
   - **SMOTE (Synthetic Minority Over-sampling Technique)** for balancing the dataset.
   - **Class Weights** to adjust model learning.

By implementing these improvements, we can develop a **more robust and scalable spam email detection system** that minimizes false positives and false negatives effectively. 🚀

---

## **📥 Installation & Setup**
### **🔹 Prerequisites**
Ensure you have **Python 3.x** installed and required dependencies:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib nltk imblearn xgboost transformers tensorflow
```

### **🔹 Run the Project**
Clone the repository and execute the script:
```bash
git clone https://github.com/your-repo/spam-email-detection.git
cd spam-email-detection
python main.py
```

## **🤝 Contributing**
Feel free to contribute to the project by submitting a **pull request** or reporting an issue! 🚀

## **📜 License**
This project is licensed under the **MIT License**.

---
### **💡 Author:**
Your Name – [GitHub Profile](https://github.com/your-profile)
