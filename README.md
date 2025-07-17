# 🟢 HeadlineHunter

**HeadlineHunter**  
_A lightweight prototype for optimizing news headlines using click-to-read prediction models._

---

## Status

- **Project health:** Actively prototyping
- **Model readiness:** Baseline working, iterative tuning ongoing
- **Last tested:** July 2025, Python 3.11, OpenAI API 1.x

---

## Overview

HeadlineHunter is an experimental framework designed to **test, optimize, and rewrite news headlines** based on predicted user engagement.  
It leverages:
- **CTR optimization models** (XGBoost, LightGBM, CatBoost)
- **LLM-based rewriting** via OpenAI GPT models
- **Similarity search** using FAISS
- **Streamlit UI** for quick testing

---

## Features

- ✅ **Streamlit Dashboard** to visualize headline performance
- ✅ **Click-to-read prediction models** (impressions-level CTR)
- ✅ **Rewriter module** for LLM-based headline rewrites
- ✅ **FAISS index** for headline similarity checks
- ✅ Modular codebase for fast experimentation

---

## 📂 Repository Structure

```plaintext
headlinehunter/
├── data/                        # Preprocessed datasets and FAISS indices
│   └── faiss_index/             # FAISS utilities and search functions
│
├── Core2/                       # EDA scripts, feature builders, backups
│   ├── EDA_indepth.py
│   ├── EDA_preprocess_features.py
│   ├── add_features.py
│   └── build_faiss_index.py
│
├── streamlit_app.py             # Main Streamlit application
├── model_loader.py              # ML model loading utilities
├── llm_rewriter.py              # LLM rewriting logic
├── feature_utils.py             # Feature engineering helpers
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation ```

---
# Clone the repository
git clone https://github.com/robertdshaw/headlinehunter.git
cd headlinehunter

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # For Windows
# source venv/bin/activate  # For Mac/Linux

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set your OpenAI API key (create a .env file)
echo OPENAI_API_KEY=sk-proj-xxxxx > .env

# Launch the Streamlit app
streamlit run streamlit_app.py
