# 👗 StyleSync: AI-Driven Personalized Outfit & Shopping Assistant

**StyleSync** is a fashion intelligence platform that predicts outfit compatibility, diagnoses mismatches, and recommends real-time shopping alternatives. Developed as a Capstone Project for the MS in Information Technology & Analytics at RIT, it combines computer vision, feature comparison, and API integrations to deliver personalized styling support.

---

![System Architecture](./screenshots/system_architecture.png)  
*Figure 1: End-to-end system pipeline including model, diagnosis, and UI layers*

---

## 🧩 Problem Statement

While many fashion platforms suggest individual clothing items, few assess how well they go together. StyleSync addresses this by:

- Taking **multiple clothing item images**
- Evaluating **outfit compatibility**
- Identifying **mismatched elements**
- Recommending **better alternatives**
- Linking users to **live shopping options**

---

## 🚀 Key Features

- ✅ Upload outfits (tops, bottoms, shoes, accessories)
- 🧠 Predict visual compatibility scores
- 📉 Diagnose mismatches with explanations
- 🛍️ Recommend better items using SerpAPI
- 🖥️ View results via a clean Dash interface

---

## 📦 Dataset Summary

We use the **Maryland Polyvore Images Dataset**, containing:

- 33,375 full outfits  
- Over 444,000 item-level images  
- Categories: Tops, Bottoms, Shoes, Bags, Accessories  
- JSON annotations for compatibility and FITB (Fill-in-the-Blank) training

---

## 🏗️ Architecture

![Model Architecture](./screenshots/mcn_architecture.png)  
*Figure 2: ResNet-50 + Multi-Layer Comparison Network (MCN) used for compatibility scoring*

### Modules Used

- **Feature Extractor:** ResNet-50  
- **Pairwise Similarity:** Layered comparison using cosine distance  
- **Diagnosis:** Gradient-based contribution mapping  
- **Inference Engine:** Cosine similarity scoring for all pairs  
- **Web Search:** Real-time shopping via SerpAPI

---

## 📊 Model Performance

- ✅ **91.2%** compatibility classification accuracy  
- 📈 **Mean Average Precision (mAP):** 0.77  
- 🧪 Binary Cross-Entropy Loss, balanced sampling for training

---

## 🧪 Diagnosis Case Studies

### Case 1: Shoes + Accessories Conflict  
**Score Before:** 0.0001 → **After Fix:** 0.9692 ✅  
| Before | After |
|--------|-------|
| ![Before](./screenshots/case1_before.png) | ![After](./screenshots/case1_after.png) |

---

### Case 2: Bottom + Shoes Clash  
**Score Before:** 0.2246 → **After Fix:** 0.9676 ✅  
| Before | After |
|--------|-------|
| ![Before](./screenshots/case2_before.png) | ![After](./screenshots/case2_after.png) |

---

### Case 3: Perfect Outfit  
**Score:** 0.9999 – No changes required  
![Perfect](./screenshots/perfect_outfit.png)

---

## 🧑‍💻 User Interface

The Dash-powered UI allows users to:

1. Upload clothing images  
2. Click “Check Compatibility”  
3. View score + improvement suggestions  
4. See smart product alternatives  

![Upload Screen](./screenshots/ui_upload.png)  
![Diagnosis Output](./screenshots/ui_diagnosis.png)  
![Web Recommendations](./screenshots/web_recommendations.png)

---

## 🌐 SerpAPI Integration

- 🔍 Extracts category-level issues
- 🌐 Queries real-time shopping items
- 🛒 Shows results with product names, images, and links
- 🤖 Optimized for quick fashion replacement options

---

## 🧰 Run StyleSync Locally

```bash
git clone https://github.com/parthgawande/StyleSync
cd StyleSync

# Install dependencies
pip install -r requirements.txt

# Set your SerpAPI key
export SERPAPI_API_KEY=your_key_here

# Launch the app
python app/main.py
