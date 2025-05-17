# 💻 AI-Powered Laptop Recommender

Welcome to the **AI-Powered Laptop Recommender** — a smart tool that helps you choose the right laptop based on your needs using natural language!

Built with **Streamlit**, **Ollama**, and **LLMs (Llama 3)**, this app refines your queries, searches semantically, and even chats with you to guide your decision. Whether you're looking for a gaming rig, a student laptop, or a workhorse for data science — just ask!

---

## ✨ Key Features

- 🔎 **Search by Plain English**  
  Ask things like: *"budget laptop for college coding under $800"*.

- 🧠 **Smart Query Understanding**  
  Your queries are refined using a local LLM (Llama 3) for better accuracy.

- 🔗 **Semantic Search with RAG**  
  Uses vector embeddings from `nomic-embed-text` to match your intent with relevant products.

- 💬 **AI Chat Assistant**  
  Ask follow-up questions or get more details from the AI assistant, who knows your context.

- 📝 **Better Product Descriptions**  
  Generate clearer, more engaging laptop descriptions with Llama 3.

- 🔐 **Runs Locally & Privately**  
  No cloud needed! Powered by Ollama for offline use.

- ⚙️ **Debug Mode**  
  See how your query is processed behind the scenes.

---

## 🗂️ Project Structure

```
product_recommendation_app/
├── app.py                # Streamlit UI
├── config.py             # App configuration
├── data_loader.py        # Loads laptop data
├── embedding_utils.py    # Embedding logic
├── llm_services.py       # LLM queries & generation
├── retrieval.py          # Semantic search logic
├── LaptrackPhase2.csv  # Your laptop dataset
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** – Web app UI
- **lama3-8b-8192** – Language model for refinement and chat
- **all-MiniLM-L6-v2** – Embedding model
- **Pandas** – Data handling

---

## ⚙️ Setup Instructions

### 🔧 Prerequisites
- [Python 3.8+](https://www.python.org/downloads/)

### 📥 Installation Steps

1. **Clone or Prepare the Project**

```bash
cd path/to/Laptop_Recommender
```

2. **Create and Activate Virtual Environment (Recommended)**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scriptsctivate
```

3. **Install Dependencies**

```bash
pip install streamlit pandas ollama
```

4. **Download Ollama Models**

```bash
ollama pull nomic-embed-text
ollama pull llama3  # Or: ollama pull llama3:8b-instruct
```

5. **Add Your Product Data**

Place `LaptrackPhase2.csv` (containing **only laptop** data) in the root folder. Make sure columns match what’s defined in `config.py`.

---

## 🚀 Run the App

1. Start Ollama in the background  
2. In terminal:

```bash
cd Laptop_Recommender
source venv/bin/activate
streamlit run app.py
```

This will open the app in your browser 🎉

---

## 🔁 How It Works (Simplified)

1. **Load Data**  
   Reads and embeds laptop data from your CSV.

2. **User Query**  
   Type what you're looking for in plain language.

3. **Smart Refinement**  
   LLM rewrites your query for better accuracy.

4. **Semantic Search**  
   Retrieves the most relevant laptops using vector similarity.

5. **Chat & Follow-up**  
   Ask more about results — the AI knows what it's showing you!

6. **Enhanced Descriptions**  
   Click to generate detailed descriptions for any laptop.

---



**📸 Screenshots / Demo**

![image alt](https://github.com/Abhiiii25/Laptop-Recommender/blob/87655cf170537146352d8d03a7f5bbd52dbae41e/Screenshot%202025-05-17%20092440.png)

![image alt](https://github.com/Abhiiii25/Laptop-Recommender/blob/87655cf170537146352d8d03a7f5bbd52dbae41e/Screenshot%202025-05-17%20092507.png)




![image alt](https://github.com/Abhiiii25/Laptop-Recommender/blob/d090cfc3f6be80dec9a390bac566e7e2281a0030/Screenshot%202025-05-17%20014513.png)



---

## 🚧 Future Enhancements

- 🧠 Add persistent vector DB (e.g., FAISS, ChromaDB)
- 🛒 Support multiple product categories
- 🔍 Advanced filters (brand, specs, price range)
- 👤 User accounts & saved history
- 🧪 Fine-tune models for better product matching
- ⚖️ Compare multiple laptops via AI chat

---

## 🙌 Final Notes

This is a privacy-first, fully local AI recommender designed for **laptop shopping with natural language**. It’s a fun blend of cutting-edge LLM tech and a simple UI.

---

🧠 *Enjoy using the AI-Powered Laptop Recommender!* 💬  
Let me know if you need help setting it up or extending it.


