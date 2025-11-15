<p align="center">
  <img src="assets/logo.png" width="220">
</p>

# 🩺 SkinDiseaseAI  

# 🩺 SkinDiseaseAI  
AI-powered skin disease detection with FastAPI & TensorFlow

[![CI](https://github.com/chiefcode77/SkinDiseaseAI/actions/workflows/ci.yml/badge.svg)](https://github.com/chiefcode77/SkinDiseaseAI/actions)

SkinDiseaseAI is a full-stack project for detecting common skin conditions from images.  
The backend is implemented with **FastAPI**, **TensorFlow**, and a **background model reload system** that allows new ML models to be deployed without restarting the server.

A Flutter-based mobile app will provide camera capture, uploading, and viewing AI predictions.

---

# 🚀 Features

### 🔍 Machine Learning Inference
- Real model or fallback dummy model  
- TensorFlow SavedModel loader  
- Image preprocessing (resize, normalize)  
- Predicts 3 classes: *benign*, *malignant*, *other*  

### ⚙️ Admin Model Reload System (Hot Reload)
- Non-blocking reload via:
  - `POST /admin/reload_model`  
  - Returns a **job ID**
- Background thread loads a new model safely  
- Track status via:
  - `GET /admin/reload_status/{job_id}`
- Check current model state:
  - `GET /admin/model_status`

### 🧪 Full Testing Support
- Unit tests for prediction and admin reload  
- Fake model injection for fast tests  
- GitHub Actions CI triggered on PRs and pushes  
- Cached dependencies for fast builds  

### 📱 Mobile App (coming soon)
- Flutter-based Android/iOS app  
- Camera capture  
- Upload to backend  
- Display risk classification  
- Model explainability (Grad-CAM) planned  

---

# 📁 Project Structure


---

# 🧰 Requirements

- Python **3.11**  
- pip  
- (Optional) TensorFlow SavedModel exported to:  
  `backend/model/saved_model/`

---

# 🛠️ Installation & Setup

## 1) Clone the repository

```bash
git clone https://github.com/chiefcode77/SkinDiseaseAI.git
cd SkinDiseaseAI/backend
