# 👗 StyleSync: AI-Driven Personalized Outfit & Shopping Assistant

StyleSync is an AI-powered fashion recommendation system that evaluates outfit compatibility, diagnoses mismatches, and suggests real-time shopping alternatives using SerpAPI.

![System Architecture](./images/system_architecture.png)
*Figure 1: High-level system architecture showing UI, model, diagnosis, and API layers*

---

## 🧠 Overview

Fashion is subjective. Traditional recommendation engines struggle with outfit cohesion. StyleSync solves this by:

- Accepting **multiple item images**
- Predicting **outfit compatibility score**
- Diagnosing weak elements
- Providing **smart improvements + shopping options**

---

## ✨ Key Features

✅ Visual and textual diagnosis  
✅ Real-time web-based shopping suggestions  
✅ Semantic outfit compatibility scoring  
✅ Feedback loop for future personalization  

![UI Upload Screen](./images/ui_upload_outfit.png)
*Figure 2: UI for uploading clothing items*

---

## 🧾 Dataset

- 33,375 outfits from Maryland Polyvore Images Dataset  
- 444,000+ item images across:
  - Tops (Upper)
  - Bottoms
  - Shoes
  - Bags
  - Accessories  
- Annotations available for category mapping and split strategy

---

## 🧱 Architecture

![Multi-Level Comparison Workflow](./images/fitb_interface_example.png)
*Figure 3: Feature extraction, pairwise similarity, scoring, and diagnosis using MCN*

**Model:**  
- ResNet-50 backbone  
- Multi-layer comparison (Layers 1–4)  
- Cosine similarity  
- Binary Cross-Entropy Loss  

---

## 🧪 Diagnosis Examples

### 🟠 Case 1: Shoe + Accessory Conflict  
**Score Before:** 0.0001  
**Fix:** Replace with boots and a casual accessory  
**Score After:** 0.9692 ✅

| Before | After |
|--------|-------|
| ![Before](./images/case1_shoe_accessory_before.png) | ![After](./images/case1_shoe_accessory_after.png) |

---

### 🔵 Case 2: Bottom + Shoe Clash  
**Score Before:** 0.2246  
**Fix:** Printed skirt + casual boots  
**Score After:** 0.9676 ✅

| Before | After |
|--------|-------|
| ![Before](./images/case2_bottom_shoe_before.png) | ![After](./images/case2_bottom_shoe_after.png) |

---

### 🟢 Case 3: Perfect Outfit  
**Score:** 0.9999  
**Diagnosis:** No conflict detected — stylistically perfect

![Perfect Outfit](./images/case3_perfect_outfit.png)
*Figure 4: Naturally compatible outfit with no recommended changes*

---

## 🛍️ Smart Shopping Suggestions

StyleSync uses **SerpAPI** to find real-world product recommendations.

![Web Suggestions](./images/ui_web_suggestions.png)
*Figure 5: Web-based product recommendations (image, title, link)*

---

## 🧑‍💻 User Interface

- Upload images → check compatibility → improve outfit → shop instantly
- Intuitive Dash interface

![Diagnosis Output](./images/ui_diagnosis_output.png)
*Figure 6: Diagnosis view showing low-score outfit and improvement options*

---

## 🔭 Future Work

- GPT-4-based free-text outfit suggestions  
- Smart occasion-based filtering (e.g., wedding, brunch)  
- Personalized fine-tuned LLMs  
- Feedback-powered active learning  
- Multimodal matching using tags + reviews  

---

## 👤 Author

**Parth Keyur Gawande**  
🎓 RIT | MS Data Science  
🔗 [LinkedIn](https://www.linkedin.com/in/parthgawande)  
🌐 [Portfolio](https://parthgawande.github.io/Portfolio)

---

## 📜 License

MIT License — free to use with credit.

---

## ⭐ Star the project if you found it useful!
