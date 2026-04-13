# agent_layer
# 🛡️ AI-Powered Secure Banking Agent

An intelligent banking simulation system that allows users to transfer money using **natural language commands**, powered by an **AI agent**, secured with a **machine learning risk detection layer**, and backed by a simple **Excel-based database**.

---

## 🚀 Features

* 🤖 **AI Agent Interface**

  * Understands natural language commands like:

    > "Send 500 rupees to account 1002"

* 🧠 **LLM-Based Decision Making**

  * Uses an AI model to interpret user intent and extract transaction details

* 🔐 **Authentication System**

  * Account number + password verification

* ⚠️ **ML Risk Detection Layer**

  * Detects potentially malicious or suspicious prompts before execution

* 💸 **Secure Transaction System**

  * Transfers money between accounts stored in an Excel database

* 🌐 **Full Stack Application**

  * Backend: FastAPI
  * Frontend: Streamlit

---

## 🏗️ Architecture

```
User (Natural Language)
        ↓
Streamlit Frontend
        ↓
FastAPI Backend (/agent)
        ↓
AI Agent (LLM)
        ↓
ML Risk Detection Pipeline
        ↓
Banking Logic (Excel DB)
        ↓
Response to User
```

---

## 📂 Project Structure

```
project/
│
├── main.py              # FastAPI backend
├── app.py               # Streamlit frontend
├── agent_llm.py         # AI agent (LLM logic)
├── bank.py              # Banking logic (auth + transfer)
├── orchestrator.py      # ML risk pipeline
├── accounts.xlsx        # Excel database
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If no requirements file:

```bash
pip install fastapi uvicorn streamlit requests pandas openpyxl openai
```

---

## 🔑 Setup

### Add your OpenAI API Key

In `agent_llm.py`:

```python
client = OpenAI(api_key="YOUR_API_KEY")
```

---

## ▶️ Running the Application

### 1. Start FastAPI backend

```bash
uvicorn main:app --reload
```

### 2. Start Streamlit frontend

```bash
streamlit run app.py
```

---

## 🌐 Access the Application

* Frontend UI: http://localhost:8501
* API Docs: http://localhost:8000/docs

---

## 🧪 Example Usage

Enter in UI:

```
Send 500 to account 1002
```

### System Flow:

1. AI agent extracts:

   * amount = 500
   * receiver = 1002
2. Authentication is verified
3. ML model checks risk
4. If safe → transaction executed
5. Response displayed

---

## ⚠️ Disclaimer

* This is a **demo project** for educational/hackathon purposes
* Excel database is not secure for real-world banking
* Passwords are stored in plain text (not recommended for production)

---

## 🔮 Future Improvements

* 🔐 Password hashing & secure auth
* 🧾 Transaction history & logs
* 💬 Conversational memory agent
* 📊 Dashboard & analytics
* 🌍 Deployment (Render / Railway / Vercel)
* 🧠 Multi-step AI reasoning

---

## 👨‍💻 Author

Built by [Your Name]

---

## ⭐ If you like this project

Give it a star ⭐ and share it!
