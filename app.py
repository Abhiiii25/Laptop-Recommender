#app.py

import streamlit as st
from data_loader import load_laptop_data
from retrieval import setup_retrieval, query_index
from llm_services import ask_groq
from config import filter_data, save_chat_history, load_chat_history
import base64
import pandas as pd

st.set_page_config(
    page_title="💻 GenAI Laptop Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling and dark mode toggle support
st.markdown(
    """
    <style>
    body, .main {
        background-color: #f9fafd;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
        letter-spacing: 1.2px;
    }
    .button-primary {
        background-color: #1e3c72;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        border-radius: 6px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .button-primary:hover {
        background-color: #16325c;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        body, .main {
            background-color: #121212;
            color: #e0e0e0;
        }
        .header {
            background: linear-gradient(90deg, #0b2340, #1a396f);
            color: #eee;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header">💻 GenAI Laptop Recommender 🚀</div>', unsafe_allow_html=True)

tabs = st.tabs(["🔍 Search", "📜 History", "💬 Feedback"])

# Search Tab
with tabs[0]:
    df = load_laptop_data()

    if "Description" not in df.columns:
        st.error("The dataset must contain a 'Description' column.")
    else:
        index, embeddings = setup_retrieval(df, description_column="Description")

        with st.sidebar:
            st.header("📊 Filters")
            min_price = st.number_input("Min Price", value=0, step=1000)
            max_price = st.number_input("Max Price", value=int(df['Price'].max()), step=1000)
            brands = st.multiselect("Select Brand", df['Brand'].unique())

        col1, col2 = st.columns([4, 1])
        with col1:
            user_query = st.text_input(
                "What laptop are you looking for?",
                placeholder="Example: lightweight laptop for programming under 60k",
            )
        with col2:
            submit = st.button("🔍 Search", key="search_button")

        if submit:
            if not user_query.strip():
                st.warning("Please enter a laptop requirement to proceed.")
            else:
                filtered_df = filter_data(df, min_price, max_price, brands)
                results = query_index(index, user_query, filtered_df)

                if results.empty:
                    st.warning("No matching laptops found. Try refining your query.")
                else:
                    st.markdown('<h3 style="color:#1e3c72;">🔍 Top Matches:</h3>', unsafe_allow_html=True)
                    st.dataframe(results)

                    groq_prompt = f"""
You are a laptop advisor. The user wants:

\"\"\"{user_query}\"\"\"

Here are the top matched laptops:
{results.to_string(index=False)}

Suggest the most suitable laptop with reasoning.
"""
                    st.markdown('<h3 style="color:#2a5298;">🤖 AI Suggestion:</h3>', unsafe_allow_html=True)
                    with st.spinner("Generating AI recommendation..."):
                        response = ask_groq(groq_prompt)
                    st.write(response)
                    save_chat_history(user_query, response)

        # Export Options for last results
        if 'results' in locals() and not results.empty:
            st.markdown("### 📁 Export Recommendations")
            csv = results.to_csv(index=False).encode('utf-8')
            b64_csv = base64.b64encode(csv).decode()
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="recommendations.csv">Download CSV</a>'
            st.markdown(href_csv, unsafe_allow_html=True)

            try:
                import pdfkit
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
                    tmp_html.write(results.to_html(index=False).encode('utf-8'))
                    tmp_html.flush()

                    pdf_data = pdfkit.from_file(tmp_html.name, False)

                b64_pdf = base64.b64encode(pdf_data).decode()
                href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="recommendations.pdf">Download PDF</a>'
                st.markdown(href_pdf, unsafe_allow_html=True)
            except ImportError:
                st.info("PDF export requires 'pdfkit' and wkhtmltopdf installed. CSV export is available.")

# History Tab
with tabs[1]:
    st.markdown("## 📜 Chat History")
    history_df = load_chat_history()
    if history_df.empty:
        st.info("No chat history yet.")
    else:
        st.dataframe(history_df)

# Feedback Tab
with tabs[2]:
    st.markdown("## 💬 Feedback")
    feedback = st.text_area("Share your thoughts or suggestions")
    if st.button("Submit Feedback"):
        st.success("Thanks for your feedback!")