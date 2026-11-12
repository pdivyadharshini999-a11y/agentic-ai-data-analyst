# 📊 Agentic AI Data Analyst

An autonomous AI Data Analyst web app built with **Python**, **Streamlit**, and **Google Gemini**.

## ✨ Features
- **Automated Dataset Profiling:** Extracts schemas, data types, and summary statistics.
- **Code Execution & Self-Correction:** Writes and executes pandas code locally, catching and fixing its own errors.
- **Dynamic Visualizations:** Generates custom matplotlib and seaborn charts.
- **Model Fallback:** Automatically switches between Gemini Flash models during high demand.

## 🚀 Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Add your API key to a `.env` file: `GEMINI_API_KEY="your-key-here"`
3. Run the app: `streamlit run app.py`
