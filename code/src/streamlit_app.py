import streamlit as st
import pandas as pd
import json
import time
from dotenv import load_dotenv

from main import generate_response, rag_store  # Import main functionality
import plotly.graph_objects as go

load_dotenv()

# -------------------------------
# Helper Functions
# -------------------------------
def generate_bulk_answers(df_ques, rag_estore, top_k, retrieval_type, reranker):
    output = []
    use_hybrid = retrieval_type == "Hybrid"
    use_rerank = reranker != "None"

    for i, row in df_ques.iterrows():
        question = row["question"]
        res = generate_response(
            query_input=question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank
        )
        output.append({
            "question_id": i + 1,
            "question": question,
            "answer": res["answer"],
            "article_ids": res["article_ids"],
            "retrieval_confidences": res["retrieval_confidences"],
            "token_count": res["token_count"],
            "latency": res["latency"],
            "answer_confidence": res["answer_confidence"]
        })
    return output


def download_json(data):
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    return json_str.encode("utf-8")


def gauge_chart(value, max_val, title, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': suffix, 'font': {'size': 20, 'color': '#2c3e50'}},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': '#c0392b'},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ddd",
            'steps': [
                {'range': [0, max_val * 0.33], 'color': '#fadbd8'},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': '#f5b7b1'},
                {'range': [max_val * 0.66, max_val], 'color': '#f1948a'}
            ],
        },
        title={'text': title, 'font': {'size': 16, 'color': '#2c3e50'}}
    ))
    fig.update_layout(height=230, margin=dict(l=15, r=15, t=30, b=15),
                      paper_bgcolor='#f7f7f7', font={'color': '#2c3e50'})
    return fig


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="RAG Knowledge Navigator",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp { background-color: #f7f7f7; color: #2c3e50; font-family: 'Segoe UI', Arial, sans-serif; }
.banner { text-align: center; padding: 20px; background: #c0392b; border-radius: 15px; margin-bottom: 20px; color: white; font-weight: bold; box-shadow: 0px 4px 12px rgba(0,0,0,0.2); }
.banner h1 { margin-bottom: 8px; font-size: 1.8em; }
div.stButton > button:first-child { background-color: #e74c3c; color: white; height: 45px; border-radius: 8px; font-weight: bold; transition: all 0.2s ease-in-out; }
div.stButton > button:first-child:hover { background-color: #c0392b; transform: scale(1.03); }
.answer-card { background: linear-gradient(120deg, #dff7e0, #e6f9e5); padding: 20px; border-radius: 12px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0, 128, 0, 0.2); transition: transform 0.3s, box-shadow 0.3s; }
.answer-card:hover { transform: scale(1.02); box-shadow: 0 8px 25px rgba(0, 128, 0, 0.35); }
.article-box { background: linear-gradient(120deg, #fef9e7, #f0f4f8); padding: 15px; border-radius: 10px; font-weight: bold; color: #2c3e50; box-shadow: 0 3px 10px rgba(0,0,0,0.1); transition: transform 0.2s, box-shadow 0.2s; }
.article-box:hover { transform: scale(1.01); box-shadow: 0 6px 20px rgba(0,0,0,0.15); }
.loader { display: inline-block; font-weight: bold; color: #c0392b; }
.loader span { animation: blink 1s infinite; margin-right: 2px; }
.loader span:nth-child(2) { animation-delay: 0.2s; }
.loader span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%,20% { opacity:0; } 50%,100% { opacity:1; } }
details > summary { cursor: pointer; font-weight: bold; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="banner">
    <h1> 🚀 Gemini RAG Pipeline — A Focused, Readable, and Reproducible Retrieval System</h1>
    <p>Ask questions or upload files to get AI-powered answers from your knowledge base</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Single Question", "📂 Bulk Upload"])

# -------------------------------
# TAB 1: Single Question
# -------------------------------
with tab1:
    st.subheader("Ask a Question")

    # -------------------------------
    # Top Controls
    # -------------------------------
    col_top1, col_top2, col_top3 = st.columns([1, 2, 2])

    with col_top1:
        top_k = st.number_input("Top K results", min_value=1, max_value=30, value=5, step=1)

    with col_top2:
        retrieval_type = st.selectbox(
            "Retrieval Approach",
            options=["Normal", "Hybrid - TFIDF + BM25 + Query Expansion"],
            help="Hybrid = Vector embeddings + BM25 + Query Expansion keyword matching for better coverage.",
        )

    with col_top3:
        reranker = st.selectbox(
            "Reranking",
            options=["None", "Cohere", "LLM Based", "Cross Encoder"],
            help="Select a reranker to improve ordering of retrieved documents. None disables reranking."
        )

    user_query = st.text_area("Enter your question:", height=100, placeholder="e.g. How do I verify my Wix Payments account?")

    if st.button("Get Answer", use_container_width=True):
        if user_query.strip():
            loader_text = st.empty()
            for _ in range(3):
                loader_text.markdown('<div class="loader">Fetching answer .</div>', unsafe_allow_html=True)
                time.sleep(0.3)
                loader_text.markdown('<div class="loader">Fetching answer ..</div>', unsafe_allow_html=True)
                time.sleep(0.3)
                loader_text.markdown('<div class="loader">Fetching answer ...</div>', unsafe_allow_html=True)
                time.sleep(0.3)

            result = generate_response(
                query_input=user_query,
                top_k=top_k,
                use_hybrid=(retrieval_type == "Hybrid"),
                use_rerank=(reranker != "None")
            )
            loader_text.empty()

            # -------------------------------
            # Display Answer
            # -------------------------------
            st.markdown(f"""
            <div class="answer-card">
                <h3>🧠 Answer:</h3>
                <p>{result['answer']}</p>
            </div>
            """, unsafe_allow_html=True)

            # -------------------------------
            # Circular Meters
            # -------------------------------
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.plotly_chart(gauge_chart(result['answer_confidence']*100, 100, "Answer Confidence", "%"), use_container_width=True)
            with col2:
                st.plotly_chart(gauge_chart(result['token_count'], max(result['token_count'], 200), "Token Count"), use_container_width=True)
            with col3:
                st.plotly_chart(gauge_chart(result['latency'], 10, "Latency (s)"), use_container_width=True)
            with col4:
                avg_retr_conf = round(sum(result['retrieval_confidences']) / len(result['retrieval_confidences']) * 100, 2) if result['retrieval_confidences'] else 0
                st.plotly_chart(gauge_chart(avg_retr_conf, 100, "Retrieval Confidence", "%"), use_container_width=True)

            # -------------------------------
            # Related Articles
            # -------------------------------
            with st.expander("🔗 Related Article IDs"):
                for idx, article_id in enumerate(result['article_ids']):
                    confidence = result['retrieval_confidences'][idx] if idx < len(result['retrieval_confidences']) else 1.0
                    st.markdown(f"""
                    <div class="article-box">
                        Article ID: {article_id} | Retrieval Confidence: {confidence}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a question first.")

# -------------------------------
# TAB 2: Bulk Upload
# -------------------------------
with tab2:
    st.subheader("Upload a File of Questions")

    # -------------------------------
    # Top Controls for Bulk
    # -------------------------------
    col_top1, col_top2, col_top3 = st.columns([1, 2, 2])
    with col_top1:
        top_k_bulk = st.number_input("Top K results", min_value=1, max_value=30, value=5, step=1, key="top_k_bulk")

    with col_top2:
        retrieval_type_bulk = st.selectbox(
            "Retrieval Approach",
            options=["Normal", "Hybrid - TFIDF + BM25 + Query Expansion"],
            help= "Hybrid = Vector embeddings + BM25 + Query Expansion keyword matching for better coverage.",
            key="retrieval_bulk"
        )

    with col_top3:
        reranker_bulk = st.selectbox(
            "Reranking",
            options=["None", "Cohere", "LLM Based", "Cross Encoder"],
            help="Select a reranker to improve ordering of retrieved documents. None disables reranking.",
            key="reranker_bulk"
        )

    uploaded_file = st.file_uploader("Upload a CSV or XLSX file containing a 'question' column", type=["csv", "xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

        if "question" not in df.columns:
            st.error("File must contain a 'question' column.")
        else:
            st.write("✅ File uploaded successfully.")
            st.dataframe(df)

            if st.button("Generate Answers for All", use_container_width=True):
                loader_text = st.empty()
                for _ in range(3):
                    loader_text.markdown('<div class="loader">Generating answers .</div>', unsafe_allow_html=True)
                    time.sleep(0.3)
                    loader_text.markdown('<div class="loader">Generating answers ..</div>', unsafe_allow_html=True)
                    time.sleep(0.3)
                    loader_text.markdown('<div class="loader">Generating answers ...</div>', unsafe_allow_html=True)
                    time.sleep(0.3)

                # Use the selected top_k, retrieval type, and reranker for bulk
                results = generate_bulk_answers(df, rag_store, top_k_bulk, retrieval_type_bulk, reranker_bulk)
                loader_text.empty()

                simplified_results = [
                    {
                        "question_id": item["question_id"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "article_ids": item["article_ids"]
                    }
                    for item in results
                ]

                st.success("✅ All answers generated!")
                display_df = pd.DataFrame(simplified_results)
                st.dataframe(display_df[["question_id", "question", "answer", "article_ids"]])

                json_data = json.dumps(simplified_results, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data.encode("utf-8"),
                    file_name="answers.json",
                    mime="application/json",
                    use_container_width=True
                )

