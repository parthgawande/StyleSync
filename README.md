# 👗 StyleSync: AI-Driven Personalized Outfit & Shopping Assistant

StyleSync is an AI-powered fashion recommendation system that analyzes outfit combinations for compatibility, identifies mismatches, and suggests real-time shopping alternatives using SerpAPI. Built as a Capstone Project for the MS in Information Technology & Analytics at RIT, StyleSync blends computer vision, data science, and API integration into a unified smart wardrobe assistant.

---

![Architecture](./screenshots/system_architecture.png)  
*Figure 1: Complete architecture overview — from user inputs to recommendations*

---

## 🔍 Problem Statement

Outfit matching is inherently subjective and lacks structured computational guidance. Many existing systems recommend individual clothing items but fail to assess **overall outfit compatibility**. StyleSync bridges this gap by:

- Accepting **user-uploaded images**
- Analyzing **outfit combinations**
- Diagnosing **incompatible items**
- Suggesting **smart replacements**
- Recommending **real-time shopping options**

---

## 🧠 Core Features

✅ Upload outfits as images  
✅ Predict outfit compatibility score  
✅ Perform diagnosis with explanations  
✅ Suggest improvements from trained knowledge  
✅ Retrieve live product alternatives via **SerpAPI**  
✅ View all recommendations on a dynamic UI  

---

## 📁 Dataset Overview

We used the **Maryland Polyvore Images Dataset**, consisting of:

- 📦 33,375 labeled outfits  
- 🖼️ 444,000+ individual item images  
- 🧵 Categories: Tops, Bottoms, Shoes, Accessories, Bags  
- 📘 JSON annotation files for compatibility, item IDs, and FITB task  

Category mapping and subsetting were performed for GPU-efficient training.

---

## 🧱 System Architecture

![Model Workflow](./screenshots/mcn_architecture.png)  
*Figure 2: ResNet-50 backbone with Multi-Layer Comparison Network (MCN)*

**Key Components:**

- 🔍 **Feature Extraction:** ResNet-50 pretrained on ImageNet  
- 🧠 **Compatibility Prediction:** Cosine similarity + Binary Cross-Entropy Loss  
- 🔁 **Comparison Model:** Multi-layered pairwise item feature matching  
- 🔄 **Diagnosis Engine:** Gradient-based analysis of contribution scores  
- 🔗 **SerpAPI Engine:** Retrieves real-time web suggestions  

---

## 📊 Performance Metrics

- ✅ 91.2% prediction accuracy on compatible vs incompatible outfit classification  
- 🔍 Mean Average Precision (mAP): **0.77**  
- 🧪 Used Cosine Similarity for final inference and diagnosis scoring  

---

## 🧪 Diagnosis Case Studies

### 🟠 Conflict Detected: Shoes + Accessories  
- **Before Score:** 0.0001  
- **Fix:** Swap shoes + accessories  
- **After Score:** 0.9692 ✅  

| Before | After |
|--------|-------|
| ![Before](./screenshots/case1_before.png) | ![After](./screenshots/case1_after.png) |

---

### 🔵 Conflict Detected: Bottom + Shoes  
- **Before Score:** 0.2246  
- **Fix:** Replace bottom with print skirt  
- **After Score:** 0.9676 ✅  

| Before | After |
|--------|-------|
| ![Before](./screenshots/case2_before.png) | ![After](./screenshots/case2_after.png) |

---

### 🟢 No Conflict Detected: Compatible Outfit  
- **Score:** 0.9999  
- **Action:** No changes suggested  

![Perfect Outfit](./screenshots/perfect_outfit.png)  
*Figure 3: Example of a naturally compatible outfit*

---

## 🧑‍💻 User Interface

StyleSync is powered by a **Dash** interface for simplicity:

- Step 1: Upload images (Top, Bottom, Shoes, Accessories)  
- Step 2: Click 'Check Compatibility'  
- Step 3: View Score + Diagnosis  
- Step 4: Get Smart Recommendations  

![UI Upload](./screenshots/ui_upload.png)  
*Figure 4: Upload images screen*

![UI Diagnosis](./screenshots/ui_diagnosis.png)  
*Figure 5: Real-time diagnosis output*

![Web Suggestions](./screenshots/web_recommendations.png)  
*Figure 6: Product search using SerpAPI*

---

## 🛍️ SerpAPI Integration

- 🔎 Extracts item categories from diagnosis
- 🌐 Queries real-world products from the web
- 📸 Shows images, titles, and direct shopping links
- 🤖 Personalized suggestions for better match

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/parthgawande/StyleSync
cd StyleSync

# Install required dependencies
pip install -r requirements.txt

# Set up SerpAPI key
export SERPAPI_API_KEY=your_api_key_here

# Launch the app
python app/main.py
