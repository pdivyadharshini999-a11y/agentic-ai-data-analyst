import os
import io
import re
import time
import traceback
import contextlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load environment variables & initialize Gemini client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Agentic AI Data Analyst", page_icon="📊", layout="wide")

if not api_key:
    st.error("GEMINI_API_KEY not found! Please add it to your .env file.")
    st.stop()

# Set a 20-second HTTP timeout so requests never hang forever
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=20000),
)

# Gather only fast Flash models (excluding image/audio/deprecated 2.x)
available_models = [
    m.name.replace("models/", "")
    for m in client.models.list()
    if "gemini" in m.name.lower()
    and "flash" in m.name.lower()
    and "image" not in m.name.lower()
    and "audio" not in m.name.lower()
    and "tts" not in m.name.lower()
    and "2.5" not in m.name.lower()
    and "2.0" not in m.name.lower()
]
if not available_models:
    available_models = ["gemini-3.8-flash"]

MODEL_ID = available_models[-1]


def generate_with_fallback(contents, config, status_box=None):
    """Tries fast Flash models with live status updates."""
    models_to_try = list(dict.fromkeys([MODEL_ID] + available_models[::-1]))[:3]
    last_error = None

    for model_name in models_to_try:
        if status_box:
            status_box.write(f"🔄 Calling `{model_name}`...")
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_error = e
            if status_box:
                status_box.write(f"⚠️ `{model_name}` busy, switching model...")
            time.sleep(1)
    raise last_error


# 2. Helper Functions for the Agent
def get_dataframe_profile(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    return (
        f"Dataset Shape: {df.shape} rows, {df.shape} columns\n\n"
        f"Column Info & Types:\n{info_str}\n\n"
        f"First 5 Rows:\n{df.head(5).to_markdown(index=False)}\n\n"
        f"Summary Statistics (Numeric):\n{df.describe().to_markdown()}\n"
    )

def extract_python_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def execute_agent_code(code: str, df: pd.DataFrame, chart_path: str = "temp_chart.png"):
    if os.path.exists(chart_path):
        os.remove(chart_path)

    local_env = {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns}
    stdout_capture = io.StringIO()
    plt.clf()
    plt.close("all")

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {}, local_env)
        return True, stdout_capture.getvalue(), os.path.exists(chart_path)
    except Exception:
        return False, traceback.format_exc(), False

def run_analyst_agent(user_question: str, df: pd.DataFrame, df_profile: str, status_box=None, max_retries: int = 3):
    system_instruction = (
        "You are an expert Senior Data Analyst AI Agent. You are given a pandas DataFrame named `df` already loaded in memory.\n"
        "Write valid Python code inside a ```python ... ``` block to analyze `df`.\n"
        "1. `df`, `pd`, `np`, `plt`, and `sns` are ALREADY imported. Do NOT load CSV/Excel files.\n"
        "2. Use `print()` to output exact numbers and tables.\n"
        "3. If a chart helps answer the question, save a matplotlib/seaborn figure strictly as 'temp_chart.png' "
        "using plt.savefig('temp_chart.png', bbox_inches='tight', dpi=150). Never call plt.show()."
    )

    conversation = [
        f"Dataset `df` profile:\n{df_profile}\n\nUser Question: {user_question}\n\nWrite Python code to answer this."
    ]

    executed_code = ""
    execution_output = ""
    chart_created = False

    # STEP A: Code Generation & Self-Correction
    for attempt in range(1, max_retries + 1):
        if status_box:
            status_box.write(f"🧠 Writing analysis code (Attempt {attempt})...")
        response = generate_with_fallback(
            contents="\n\n".join(conversation),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
            status_box=status_box,
        )
        
        ai_text = response.text or ""
        code = extract_python_code(ai_text)
        if not code:
            return ai_text, None, None

        if status_box:
            status_box.write("⚙️ Running Python analysis on your dataset...")
        executed_code = code
        success, result, chart_created = execute_agent_code(code, df)

        if success:
            execution_output = result if result.strip() else "Code executed successfully."
            break
        else:
            if status_box:
                status_box.write("🛠️ Fixing a minor code issue and retrying...")
            conversation.append(f"Assistant Code:\n{code}")
            conversation.append(f"Error:\n{result}\n\nFix the bug and return corrected ```python``` code.")
            if attempt == max_retries:
                return f"Could not complete code execution after {max_retries} attempts:\n\n{result}", executed_code, None

    # STEP B: Final Insight Synthesis
    if status_box:
        status_box.write("✍️ Summarizing insights...")
    chart_status = "Yes (shown to user)" if chart_created else "No"
    synthesis_prompt = (
        f"User Question: {user_question}\n\n"
        f"Execution Output:\n{execution_output}\n\n"
        f"Chart Generated: {chart_status}\n\n"
        "Provide a clear, concise data analyst summary based ONLY on the output above. Highlight key numbers in bold."
    )

    final_response = generate_with_fallback(
        contents=synthesis_prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        status_box=status_box,
    )

    chart_bytes = None
    if chart_created and os.path.exists("temp_chart.png"):
        with open("temp_chart.png", "rb") as f:
            chart_bytes = f.read()

    return final_response.text, executed_code, chart_bytes

# 3. Streamlit User Interface
st.title("📊 Agentic AI Data Analyst")
st.caption(f"Primary model: `{MODEL_ID}` | Fallbacks ready: `{', '.join(available_models[:3])}`")

with st.sidebar:
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if st.button("Clear Chat History"):
        st.session_state.messages = list()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = list()

if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.session_state.current_file = uploaded_file.name
        st.session_state.df_profile = get_dataframe_profile(st.session_state.df)
        st.session_state.messages = list()

    df = st.session_state.df

    with st.expander("🔍 Preview Dataset & Summary", expanded=False):
        st.write(f"**Rows:** {df.shape} | **Columns:** {df.shape}")
        st.dataframe(df.head(10), use_container_width=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart"):
                st.image(msg["chart"])
            if msg.get("code"):
                with st.expander("🛠️ View Agent's Python Code"):
                    st.code(msg["code"], language="python")

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("🤖 Agent is working...", expanded=True) as status:
                answer, code_used, chart_img = run_analyst_agent(
                    prompt, df, st.session_state.df_profile, status_box=status
                )
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            st.markdown(answer)
            if chart_img:
                st.image(chart_img)
            if code_used:
                with st.expander("🛠️ View Agent's Python Code"):
                    st.code(code_used, language="python")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "code": code_used,
            "chart": chart_img
        })
else:
    st.info("👈 Upload a `.csv` or `.xlsx` file in the sidebar to wake up your AI Data Analyst!")
