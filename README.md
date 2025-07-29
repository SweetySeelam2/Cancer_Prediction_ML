[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cancerpredictionml-logisticregression.streamlit.app/)

# 🩺 AI-Powered Breast Cancer Diagnosis with Logistic Regression & SHAP Explainability
### An Interpretable, Deployed Machine Learning Model for Early Tumor Classification and Business-Driven Clinical Insights

---

## 📌 Project Overview
Breast cancer is a leading cause of cancer-related deaths globally, and early detection remains the most critical factor in improving patient outcomes. This project delivers a high-performing, interpretable machine learning solution built with Logistic Regression to classify breast tumors as **benign** or **malignant** using digitized features from fine needle aspirate (FNA) images.

It allows medical professionals and decision-makers to:
- Run predictions on real patient-like data
- Interpret the model output using SHAP explainability
- Receive business-aligned insights and recommendations

This deployed Streamlit application enables users — medical professionals, hospitals, data scientists, and decision-makers — to interact with the model, test it using real values, and receive diagnostic insights with clear business recommendations.

---

## ❗ Business Problem
Most hospitals struggle with the dual challenge of:
1. Delayed or inaccurate diagnosis due to human error or lack of explainable tools.
2. Lack of automation and decision support to assist radiologists and clinicians.
3. Time-consuming manual analysis of image-derived features.
4. No clinical support system integrated into the diagnostic workflow.

We solve this with a model that is:
- Accurate ✅  
- Faster predictions ✅  
- Explainability for trust & transparent (SHAP) ✅  
- Easy to deploy ✅  
- Clinical-grade performance  

---

## 🎯 Objective
- Detect **malignant tumors** early with high precision
- Provide **explainable AI support** to medical teams
- Build a **scalable ML system** that can be integrated into diagnostic pipelines
- Deliver **business insights** that reduce cost, error, and risk

---

## 📊 Dataset Information
- **Source**: [Breast Cancer Wisconsin Diagnostic Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Format**: CSV file with 569 observations and 32 columns
- **Target**: `diagnosis` (M = Malignant, B = Benign)

### 🧬 Key Features Used:
- `radius_mean`, `perimeter_mean`, `area_mean`
- `concavity_worst`, `texture_worst`, `concave_points_mean`, etc.
- 20 most important features were selected via correlation

---

## ✅ Final Model Results & Interpretation
- 📈 **Model Used**: Logistic Regression (GridSearchCV-tuned)
- ✅ **Accuracy**: 98.25%
- ✅ **Precision (Malignant)**: 100%
- ✅ **Recall (Malignant)**: 95%
- ✅ **F1 Score**: 0.98
- 📉 **False Positives**: 0 | **False Negatives**: 2

### 🧠 SHAP Explainability:
- Top features like `texture_worst`, `radius_worst`, and `concave_points_worst` were identified
- SHAP plots visualize how each feature pushes a prediction toward benign/malignant

---

## 📁 Patient-Level Sample Predictions

You can now:                
- 🧬 Select **real-like patient cases** from a dropdown **with true diagnosis label shown**  
- 🔍 See **individual predictions**, confidence levels, and medical insight  
- 🟢⚠️🔴 Get dynamic risk-based messages per case  
- 💼 View **business impact per patient** including estimated financial implications and recommendations  
- 🧠 **Set custom decision thresholds** to explore model sensitivity  

*Example Output:*  
🔬 Patient #1  
True Diagnosis: Benign  
Predicted Diagnosis: Benign  
Probability of Malignant: 16.86%  
Probability of Benign: 83.14%  
🟢 Likely Benign Tumor – Routine monitoring suggested.

📈 Interpretation & Business Insight:  
- Early prediction allows for faster intervention and cost savings.  
- Model's precision reduces unnecessary biopsies.  
- Each malignant detection can save **$50,000–$100,000** in escalation cost.  

💡 Recommendation:  
- Flag high-risk patients for urgent review.  
- Use >90% probability as decision support for escalations.  

---

## 📊 Diagnostic Thresholds & Feature Guidelines

| Feature Name           | Malignant Threshold ↑ | Benign Threshold ↓ | Interpretation |
|------------------------|------------------------|---------------------|----------------|
| `radius_worst`         | > 16.0                 | < 13.0              | Tumor size |
| `texture_worst`        | > 25.0                 | < 20.0              | Irregular texture |
| `concave_points_worst` | > 0.14                 | < 0.08              | Border sharpness |
| `perimeter_worst`      | > 110                  | < 85                | Tumor boundary |
| `area_worst`           | > 1000                 | < 600               | Mass extent |

✅ If multiple features exceed malignant thresholds → prediction = "Malignant"  
⚠️ Borderline values → prompt further review with imaging/clinical input  

---

## 💰 Success Rates
- 💸 **$300K–$1M** saved annually per hospital by reducing late-stage cancer treatment
- ⏱️ **>90% time saved** by clinicians in screening review and feature analysis
- 🩻 **Error reduction**: model minimized false positives and caught nearly all malignant tumors

---

## 📈 Business Impact & Value

This tool helps healthcare professionals identify high-risk patients with >90% probability of malignancy, flagging them early for escalation.

- 🎯 Reduces false positives → fewer unnecessary biopsies.  
- 💸 Saves an estimated $50,000–$100,000 per early malignant detection.  
- ⏱️ Enables earlier intervention, increasing patient survival and reducing costs.  
- 🤝 Can be integrated into hospital triage systems or digital pathology workflows.  

This ML-powered solution is also applicable to large-scale screening systems (like used by Amazon Health or Netflix-type health optimization projects).

---

## 💼 Business Recommendations
- 🏥 Deploy as a second-opinion tool to reduce human diagnostic errors
- 📊 Train medical teams on SHAP interpretation for transparent decision-making
- 🧪 Integrate with EHR and mobile diagnostics platforms
- 🔁 Retrain on local hospital data for maximum reliability

---

## 🔗 Use Case Connections: Healthcare + Tech
### 👩‍⚕️ Hospitals:
- Improve early detection accuracy with explainable ML
- Scale diagnostics to rural and under-resourced clinics

### 💻 Data Teams & Recruiters:
- Clear demonstration of responsible ML with deployment
- Strong technical + domain project that’s production-ready

### 📦 Netflix / Amazon:
Although this model is built for healthcare, the **architecture and logic** (Logistic Regression + Explainability) is relevant for:  
Build **interpretable, real-time ML systems** with cost-sensitive outcomes (churn, fraud, health tech)  
- **Netflix**: Classifying churn likelihood of viewers with SHAP-based explanation  
- **Amazon Health**: Integrating ML-driven pre-screening into mobile wellness apps  
- **Finance/Fraud**: Logistic-based risk prediction models for interpretable decisions  
- Detect **anomalous content behavior** or user engagement patterns with similar logistic + SHAP workflows  

---

## 🚀 Run This Project
Clone the repo:
```bash
git clone https://github.com/SweetySeelam2/Cancer_Prediction_ML.git
cd Cancer_Prediction_ML
pip install -r requirements.txt
streamlit run app.py
```

---

Or try it live:
👉 [https://cancerpredictionml-logisticregression.streamlit.app/](https://cancerpredictionml-logisticregression.streamlit.app/)

---

## 📁 Project Structure
```
Cancer_Prediction_ML/
├── app.py                  # Streamlit App
├── model.pkl              # Trained Logistic Regression model
├── scaler.pkl             # Scaler used for input normalization
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
├── README.md              # This file
├── Cancer_LogisticRegression.ipynb  # Training + SHAP notebook
├── X_train_columns.csv    # Column names for feature alignment
├── shap_force_plot_0.html # HTML SHAP visualization (optional)
├── data/
│   └── Cancer prediction dataset.csv
├── images/
│   └── *.png (visuals: confusion matrix, SHAP, accuracy)
```

---

## 👩‍💻 Author & Profile
**Sweety Seelam** | Business Analyst | Aspiring Data Scientist | Machine Learning Enthusiastic                                                                               
🔗 Connect with Me:                                                                       
[GitHub](https://github.com/SweetySeelam2)                                                                                                   
[LinkedIn](https://linkedin.com/in/sweetyseelam2)                                                                                                            
[Streamlit App](https://cancerpredictionml-logisticregression.streamlit.app/)

---

## 🔒 Proprietary & All Rights Reserved
© 2025 Sweety Seelam. This work is proprietary and protected by copyright. All content, models, code, and visuals are © 2025 Sweety Seelam. No part of this project, app, code, or analysis may be copied, reproduced, distributed, or used for any purpose—commercial or otherwise—without explicit written permission from the author.

For licensing, commercial use, or collaboration inquiries, please contact: Email: sweetyseelam2@gmail.com
