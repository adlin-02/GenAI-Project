import streamlit as st
import requests
import pandas as pd

# Page setup
st.set_page_config(page_title="Research Article Summarizer", layout="centered")
st.title("📄 Research Article Summarizer (Groq + LLaMA3)")

# Input: Groq API Key and Article Text
groq_api_key = st.text_input("🔑 Enter your Groq API Key", type="password")
article_text = st.text_area("📚 Paste your research article here (500–1000 words)", height=300)

# Mode Selection: Basic or Advanced
mode = st.radio("Select Prompting Mode", ["Basic Prompting", "Advanced Prompting"])

# Prompt Type Selection
if mode == "Basic Prompting":
    prompt_type = st.radio("Choose Basic Prompt Type", ["Zero-Shot", "One-Shot", "Few-Shot"])
else:
    prompt_type = st.selectbox("Choose Advanced Prompting Technique", [
        "Chain-of-Thought", "Tree-of-Thought", "Role-Based",
        "ReAct", "Directional Stimulus", "Step-Back"
    ])

# Store summaries
if "summary_log" not in st.session_state:
    st.session_state["summary_log"] = pd.DataFrame(columns=["Technique", "Summary"])

# Summarization
if st.button("Summarize") and groq_api_key and article_text:
    with st.spinner("Generating summary..."):

        # Basic Prompting
        if mode == "Basic Prompting":
            if prompt_type == "Zero-Shot":
                prompt = f"""
You are a scientific assistant. Summarize the following research article in **1 paragraph**, not exceeding 150 words. Your summary must include:
- Background introduction
- Research objective
- Methodology
- Key results
- Final takeaway

Article:
\"\"\"{article_text}\"\"\"
"""
            elif prompt_type == "One-Shot":
                prompt = f"""
You are a scientific assistant. Summarize the following research article in **1 paragraph**, not exceeding 150 words. Your summary must include:
- Background introduction
- Research objective
- Methodology
- Key results
- Final takeaway

Example:
Article:
\"\"\"Recent advances in cancer immunotherapy...\"\"\"
Summary:
Recent developments in cancer immunotherapy have shown promise...

Now summarize this article:
Article:
\"\"\"{article_text}\"\"\"
"""
            else:  # Few-Shot
                prompt = f"""
You are a scientific assistant. Summarize the following research article in **1 paragraph**, not exceeding 150 words. Your summary must include:
- Background introduction
- Research objective
- Methodology
- Key results
- Final takeaway

Example 1:
Article:
\"\"\"Recent advances in cancer immunotherapy...\"\"\"
Summary:
Recent developments in cancer immunotherapy have shown promise...

Example 2:
Article:
\"\"\"Climate change is affecting agricultural productivity globally...\"\"\"
Summary:
This study investigates the impact of climate change on wheat production...

Now summarize this article:
Article:
\"\"\"{article_text}\"\"\"
"""

        # Advanced Prompting
        else:
            prompt_templates = {
                "Chain-of-Thought": f"""
Summarize the article step-by-step:
1. What is the background of the study?
2. What is the objective?
3. How was the study conducted (methodology)?
4. What are the key findings?
5. What conclusion is drawn?

Article:
\"\"\"{article_text}\"\"\"
""",
                "Tree-of-Thought": f"""
Generate 3 different summary versions (focus on objective, results, and implications).
Then select the most insightful one and present it clearly.

Article:
\"\"\"{article_text}\"\"\"
""",
                "Role-Based": f"""
You are a peer reviewer for a top-tier academic journal.
Write a concise summary of the research article for the editorial board.

Article:
\"\"\"{article_text}\"\"\"
""",
                "ReAct": f"""
Think step by step: first identify the objective, then list methods, then summarize key results.
If a part is unclear, mention it briefly and proceed.

Article:
\"\"\"{article_text}\"\"\"
""",
                "Directional Stimulus": f"""
Summarize the article with more emphasis on **experimental results and statistical evidence**.
Avoid vague or generic statements.

Article:
\"\"\"{article_text}\"\"\"
""",
                "Step-Back": f"""
First, reflect on this: What fundamental question is this article trying to answer?
Then give a one-paragraph summary that centers around that core insight.

Article:
\"\"\"{article_text}\"\"\"
"""
            }

            prompt = prompt_templates[prompt_type]

        # Groq API call
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300
            }
        )

        if response.status_code == 200:
            summary = response.json()["choices"][0]["message"]["content"]
            st.subheader("📝 Summary")
            st.success(summary.strip())

            # Save summary
            new_entry = pd.DataFrame([[prompt_type, summary.strip()]], columns=["Technique", "Summary"])
            st.session_state.summary_log = pd.concat([st.session_state.summary_log, new_entry], ignore_index=True)
        else:
            st.error("❌ Failed to connect to Groq API. Please check your API key and try again.")

# Show log
if not st.session_state.summary_log.empty:
    st.markdown("### 🧾 Summary Log")
    st.dataframe(st.session_state.summary_log)