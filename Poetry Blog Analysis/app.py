import os
import streamlit as st
from dotenv import load_dotenv
from evaluate_rag import evaluate_all
from poetry_rag_core import generate_poetry_analysis
import pandas as pd

load_dotenv()

st.set_page_config(page_title="📜 Poetry RAG", layout="centered")

# ---------- CUSTOM STYLES ----------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #fefefe;
    }
    .stButton>button {
        background-color: #6c5ce7;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #4c3ccc;
    }
    .stTextInput > div > div > input, .stTextArea > div > textarea {
        border: 2px solid #6c5ce7;
        border-radius: 8px;
    }
    .stMarkdown h1 {
        color: #2d3436;
    }
    .stExpanderHeader {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<h1 style='color: #6c5ce7; font-weight: 800; font-size: 2.5rem; margin-bottom:8px;'>
<span style="font-size:2.4rem;vertical-align:middle;">📜</span> Poetry Analysis <span style="color:#4c3ccc">RAG</span>
</h1>
""", unsafe_allow_html=True)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["💬 Ask AI", "📊 Evaluation Dashboard"])

# ---------- TAB 1: ASK AI ----------
with tab1:
    st.info("🤖 Ask a question about the poetic corpus")

    query = st.text_area("What would you like to ask about the poems?", height=100, placeholder="e.g., What does the poet imply about hope in document 3?")

    if st.button("✨ Generate Answer"):
        if not query.strip():
            st.warning("⚠️ Please enter a poetic question.")
        else:
            with st.spinner("⏳ Generating answer..."):
                try:
                    answer, sources = generate_poetry_analysis(query)
                    if not answer or answer.strip().lower() in ("none", "undefined", ""):
                        st.error("⚠️ No valid answer generated.")
                    else:
                        st.success("✅ Answer Generated:")
                        st.markdown(f"> {answer}")

                        if sources:
                            st.markdown("### 📚 Retrieved Context")
                            for i, chunk in enumerate(sources):
                                with st.expander(f"📄 Source {i+1}"):
                                    st.code(chunk, language="markdown")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ---------- TAB 2: EVALUATION ----------
with tab2:
    if st.button("🧪 Run Full Evaluation"):
        with st.spinner("Running evaluation on dataset..."):
            try:
                report = evaluate_all()
                if not report or "summary" not in report:
                    st.error("❌ Evaluation failed: No results returned.")
                else:
                    summary = report["summary"]
                    st.success("✅ Evaluation Completed")
                    st.markdown(f"- **Total Evaluated:** `{summary.get('total_evaluated', '?')}`")
                    st.markdown(f"- **Average F1 Score:** `{summary.get('average_f1_score', 0):.3f}`")
                    st.markdown(f"- **Average BERTScore (F1):** `{summary.get('average_bertscore_f1', 0):.3f}`")

                    evaluations = report.get("evaluations", [])
                    if not evaluations:
                        st.warning("⚠️ No evaluation entries found.")
                    else:
                        st.markdown("### 📑 Evaluation Details")
                        for i, ev in enumerate(evaluations):
                            with st.expander(f"🔎 Q{i+1}: {ev.get('query', '')[:60]}..."):
                                st.markdown(f"**📝 Query:** {ev.get('query', '')}")
                                st.markdown(f"**✅ Expected Answer:** {ev.get('expected_answer', '')}")
                                st.markdown(f"**🧠 Generated Answer:** {ev.get('generated_answer', '')}")
                                st.markdown(f"**📏 F1 Score:** `{ev.get('f1_score', 0):.3f}`")
                                st.markdown(f"**📊 BERTScore (F1):** `{ev.get('bertscore_f1', 0):.3f}`")

                    st.markdown("### 📈 Metrics Visualization")
                    metrics_data = {
                        'Metric': ['F1 Score', 'BERTScore (F1)'],
                        'Score': [summary.get('average_f1_score', 0), summary.get('average_bertscore_f1', 0)]
                    }
                    df = pd.DataFrame(metrics_data)
                    st.bar_chart(df.set_index("Metric"))
            except Exception as e:
                st.error(f"❌ Evaluation failed:\n{e}")
