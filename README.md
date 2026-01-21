![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![GAN](https://img.shields.io/badge/Model-GAN-orange.svg)
![Healthcare](https://img.shields.io/badge/Domain-Healthcare-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/asif-khan-ak/building_synthetic_medical_records/blob/main/Building_Synthetic_Medical_Records_Using_GANs.ipynb
)

# 🧬 Building Synthetic Medical Records Using GANs

---

## 📌 Project Overview

Access to high-quality medical data is often restricted due to **privacy, ethical, and regulatory concerns**.  
This project demonstrates how **Generative Adversarial Networks (GANs)** can be used to generate **realistic synthetic medical records** while preserving statistical patterns of real patient data and **mitigating privacy risks**.

The focus is on **structured tabular healthcare data**, including demographics, vitals, lab measures, lifestyle factors, and clinical outcomes.  
The GAN architecture is implemented using **PyTorch**, enabling **plausible synthetic patient follow-up records** generation.

---

## 🎯 Objectives

- **Preprocess mixed-type medical data** (numerical + categorical)  
- **Design and train a GAN** for tabular healthcare data  
- **Generate realistic synthetic medical records**  
- **Preserve statistical characteristics** without copying real patients  
- **Demonstrate privacy-preserving data generation** for healthcare analytics  

---

## 📂 Dataset Description

The dataset contains **longitudinal follow-up records** for patients with **34 features** across multiple clinical domains.

| Category | Features |
|----------|----------|
| **Demographics** | `patient_id`, `age_years`, `visit_date` |
| **Anthropometrics & Vitals** | `weight_kg`, `bmi`, `systolic_bp_mmHg`, `diastolic_bp_mmHg`, `heart_rate_bpm`, `body_temp_C` |
| **Glycemic & Lab Measures** | `fasting_glucose_mg_dL`, `postprandial_glucose_mg_dL`, `hba1c_percent` |
| **Lifestyle Factors** | `diet_quality_score_0_100`, `sleep_hours`, `exercise_sessions_per_week`, `alcohol_units_per_week`, `smoking_cigs_per_day` |
| **Clinical Outcomes (Binary)** | `neuropathy`, `retinopathy`, `hypoglycemia`, `uti` |
| **Treatment & Notes** | `medications`, `clinical_notes` |

> **Note:** Dataset is **synthetic or anonymized** and intended strictly for **educational/research purposes**.

---

## 🔄 Data Preprocessing Pipeline

1. **Categorical Encoding**  
   - One-Hot Encoding using `sklearn.preprocessing.OneHotEncoder`  

2. **Numerical Scaling**  
   - Scaled to `[-1, 1]` using `MinMaxScaler` (matches Tanh activation)  

3. **Feature Fusion**  
   - Concatenated encoded categorical and scaled numerical features  
   - **Final feature dimensionality:** 60 features  

---

## 🧠 Model Architecture

### 🔹 Generator
- Maps random noise to synthetic medical records  
- **Input:** 64-dimensional latent vector  
- **Layers:** 64 → 128 → 256 → 60  
- **Activations:** ReLU, LeakyReLU, Tanh  

### 🔹 Discriminator
- Distinguishes real vs synthetic records  
- **Input:** 60-dimensional feature vector  
- **Layers:** 60 → 256 → 128 → 1  
- **Activations:** LeakyReLU, Sigmoid  

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Loss | Binary Cross-Entropy (`BCELoss`) |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Batch Size | 32 |
| Epochs | 2000 |
| Latent Dim | 64 |

---

## 📉 Training Results

```

Epoch [2000/2000]
Discriminator Loss: 0.1006
Generator Loss:     2.8885

````

- **Discriminator** effectively distinguishes real vs fake  
- **Generator** produces realistic synthetic samples  
- Stable adversarial training without mode collapse  

---

## 🧪 Synthetic Data Generation

**20 synthetic patient records** were generated.

**Postprocessing Steps:**  
- Map outputs from `[-1, 1]`  
- Inverse-transform numerical features  
- Reconstruct categorical features via inverse one-hot  
- Convert to Pandas DataFrame  

**Example Characteristics:**  
- Realistic vitals and lab measures  
- Clinically plausible glucose and HbA1c values  
- Consistent medication patterns  
- No direct replication of real patients  

---

## 🔐 Privacy & Ethical Considerations

- GANs **do not memorize individual patients**  
- Synthetic data **reduces risk of leakage**  
- Useful for:
  - Model prototyping  
  - Education & training  
  - Data augmentation  

> ⚠ **Note:** Not intended for clinical decision-making.

---

## 🧰 Tech Stack

- Python | PyTorch | Pandas | NumPy  
- Scikit-learn | Matplotlib | Seaborn  
- Google Colab  

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/building_synthetic_medical_records.git
cd building_synthetic_medical_records
pip install -r requirements.txt
````

Open and run the notebook:
`Building_Synthetic_Medical_Records_Using_GANs.ipynb`

---

## 👤 Author

**Asif Khan**
Data Science & Machine Learning Enthusiast

---

## ⭐ Acknowledgments

Inspired by research on **synthetic data generation for healthcare** and **privacy-preserving ML**.
