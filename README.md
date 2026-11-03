# PetVision: Advanced Multi-Level Pet Classification System

PetVision is a cutting-edge **Hybrid Intelligence** diagnostic platform that bridges the gap between raw computer vision and professional veterinary consultation. By integrating **Multi-Task Learning (MTL)**, **Explainable AI (XAI)**, and **Large Multimodal Models (LMM)**, it transforms simple breed identification into a transparent and actionable health-insight tool.


## 🎯 Project Vision

Traditional pet identification systems often operate as "black boxes," providing a single label with no underlying reasoning. PetVision utilizes a three-tier architecture (**Perception, Logic, and Cognition**) to resolve the *"Semantic Gap,"* ensuring biological consistency and building user trust through visual evidence.


## 🏗️ The Three-Tier Architecture

1. **Perception Layer (CNN):** A MobileNetV2-based Multi-Task Learning model trained on 146,000+ images. It handles raw visual feature extraction.
2. **Logic Layer (Guardrails):** A deterministic Python layer that cross-references CNN predictions with an `attributes.csv` knowledge base to maintain data integrity.
3. **Cognition Layer (LMM):** Powered by the **Mistral AI API (Pixtral-12B-2409)**, this layer performs "Visual Reasoning" to generate human-readable advice and handle "Non-Pet" edge cases (e.g., toys or food).


## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow, Keras (MobileNetV2 MTL)
* **Large Multimodal Models:** Mistral AI API (Pixtral-12B-2409)
* **Computer Vision:** OpenCV, Grad-CAM (Visual Explainability)
* **Frontend:** Streamlit (Reactive Framework), Custom CSS3
* **Data Engineering:** Pandas, NumPy, Regex-based ETL Pipelines


## 📂 Project Structure

```text
PetVision/
├── app.py                   # UI Controller & Event Routing
├── logic.py                 # AI Engine (TensorFlow & Grad-CAM logic)
├── language.py              # i18n Localization Dictionaries
├── attributes.csv           # Knowledge Base (280+ Breeds & Traits)
├── label_translations.json  # Bilingual Tag Mapping
├── style.css                # Custom Diagnostic-themed Styling
└── llm_utils/               # LMM Service Wrappers & Prompts
```


## 🚀 Quick Start

Follow these steps to set up and run PetVision locally:

1. **Create and activate a virtual environment:**
   ```bash
   py -3.11 -m venv venv
   venv\Scripts\activate
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit web application:**
   ```bash
   cd main_code
   python -m streamlit run app.py
   ```


## 👥 Contributors

* **Wong Ho Yin:** Team Coordinator, Logic Guardrails
* **Chan Tsz Ho:** Frontend Lead, UI/UX Architect, Software Refactoring, Data Engineering
* **Yu Yiu Pang:** LLM Backend, Prompt Engineering, Cognitive Layer
* **Huang Liming:** Machine Learning Engineer, CNN Model Training


## 🔮 Future Work

* **Clinical Marker Detection:** Automating the detection of dermatological and ocular health markers.
* **RAG Knowledge Base:** Connecting the LLM to a Retrieval-Augmented Generation (RAG) system based on professional veterinary textbooks.
* **Scale Calibration:** Utilizing depth-sensing data or reference markers to resolve *"Scale Ambiguity"* in 2D imagery.
