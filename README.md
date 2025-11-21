# Chatur AI – Intelligent RAG Study Companion

A full-stack **AI-powered learning assistant** built using **FastAPI**, **Groq LLMs**, **MongoDB Vector Search**, and a modern **TailwindCSS frontend**.  
Chatur AI analyzes PDFs, URLs, and Notes, generates contextual answers, summaries, flashcards, quizzes, and maintains full conversation memory per chat.

---

## Features

### 🔐 Authentication  
- Secure JWT Authentication (Register/Login)  
  (Implementation: `api.py` → `/auth/register`, `/auth/login`) :contentReference[oaicite:0]{index=0}  
- Password hashing via `passlib`  
- Middleware-based token verification for protected routes

---

### 💬 Chat System  
- Multi-chat support with titles and timestamps  
- Hybrid & Strict response modes  
- Conversation history stored in MongoDB  
- Intelligent prompt construction with dynamic system instructions

---

### 📄 Document Intelligence  
Supports:  
- **PDF Upload**  
- **URL Scraping**  
- **Raw Text Notes**

Document processing handled via:  
- `DocumentProcessor` class (`app.py`) — PDF parsing, URL scraping, chunking :contentReference[oaicite:1]{index=1}  
- Text chunking with overlap for optimal embeddings  
- PDF parsing via PyPDF2 / pypdf fallback  
- BeautifulSoup4 for web content extraction  

All uploaded content is chunked → embedded → indexed.

---

### 🔎 Vector Search (RAG)  
- Embeddings generated using **SentenceTransformers MiniLM-L6-v2** (`embedder.py`) :contentReference[oaicite:2]{index=2}  
- MongoDB Atlas `$vectorSearch` retrieves the top-K relevant document chunks  
- Hybrid Mode: Mix of RAG + general knowledge  
- Strict Mode: Only use retrieved context; no hallucinations allowed

---

### 🧠 AI-Powered Tools  
Chatur AI provides a set of academic tools:

| Tool | Description |
|------|-------------|
| **Contextual Q&A** | Uses RAG + LLM reasoning |
| **Summarization** | Markdown-formatted structured summary |
| **Quiz Generator** | Multi-question quiz from documents |
| **Flashcards** | Concept → explanation flashcards |

---

### 🎨 Modern Tailwind Frontend  
Frontend built using:  
- TailwindCSS  
- Animated UI components  
- Clean message bubbles with Markdown rendering  
- File uploads, URL modals, text modals  
- Chat history, loaders, floating icons  
Source: `index.html` :contentReference[oaicite:3]{index=3}

Also includes login/register pages:  
- `login.html` :contentReference[oaicite:4]{index=4}  
- `register.html` :contentReference[oaicite:5]{index=5}

---

## 🏗️ Project Structure
📦 ChaturAI
```
├── api.py             # Main FastAPI backend (auth, chat, uploads, RAG logic)
├── app.py             # DocumentProcessor for PDF/URL/Text chunking
├── embedder.py        # MiniLM Embedding Model
├── db.py              # MongoDB connection + collections
├── models.py          # Pydantic models for Request/Response schemas
├── index.html         # Main frontend UI
├── login.html         # Login page
├── register.html      # Signup page
├── requirements.txt   # Dependencies
└── README.md
```
---

## ⚙️ Tech Stack

### Backend  
- **FastAPI**  
- **Groq LLM (llama-3.1-8b)**  
- **MongoDB Atlas + Vector Search**  
- **JWT Authentication**  
- **SentenceTransformers Embeddings**

### Frontend  
- **HTML + TailwindCSS**  
- **showdown.js** (Markdown → HTML)  
- **DOMPurify** (XSS sanitization)

### Document Processing  
- **PyPDF2 / pypdf**  
- **BeautifulSoup4**  
- **Smart chunking algorithm**

---

## 🧩 API Overview

### 🔐 Authentication  
`POST /auth/register` – Create account  
`POST /auth/login` – Obtain JWT token  

### 💬 Chat  
`POST /chats/create`  
`GET /chats/my`  
`PUT /chats/{chat_id}`  
`DELETE /chats/{chat_id}`  
`GET /chats/{chat_id}/messages`

### 📄 Upload Content  
`POST /documents/upload/pdf`  
`POST /documents/upload/url`  
`POST /documents/upload/text`

### 🤖 RAG Chat  
`POST /chat`  
- Runs RAG pipeline  
- Builds system prompt  
- Calls Groq LLM  
- Saves conversation to MongoDB  

### 🧠 Tools  
`POST /api/summarize`  
`POST /api/flashcards`  
`POST /api/quiz`

---

## 🗄️ Database Schema (MongoDB)

### Users (`users`)
{
name,
email,
password_hash,
created_at
}

### Chats (`chats`)
{
user_id,
title,
created_at
}

### Messages (`messages`)
{
chat_id,
role: "user" | "assistant",
content,
timestamp
}

### Documents (`documents`)
{
user_id,
chat_id,
title,
doc_type,
chunk_count
}

### Embeddings (`embeddings`)
{
doc_id,
chat_id,
chunk_text,
embedding[]
}

---

## 🚀 Running Locally

### 1️⃣ Install dependencies 
```
pip install -r requirements.txt
```

### 2️⃣ Set environment variables  
Create a `.env`:
```
MONGO_URI=your-mongodb-uri
MONGO_DB_NAME=chatur_ai
GROQ_API_KEY=your-api-key
JWT_SECRET=your-secret
JWT_ALGORITHM=HS256
```

### 3️⃣ Start backend  
```
uvicorn api:app --reload
```

### 4️⃣ Open frontend 
```
Open `index.html` in browser (requires backend running on port 8000).
```

---

## 🔒 Security

- All messages & documents are user-scoped  
- JWT token validation on all protected routes  
- PDF & URL sanitization  
- Vector index filtered by `chat_id` + `user_id`  
- No cross-user data access possible  

---

## 🤝 Contribution

Feel free to contribute:  
- UI improvements  
- Additional document formats  
- More academic tools  
- Multi-language support  

---

## 📄 License  
MIT License. Free for personal & academic use.

---

## 🧑‍💻 Author  
**Ansh Pradhan**  
Creator of Chatur AI – AI/ML Engineer & Python Developer  
LinkedIn: https://www.linkedin.com/in/anshpradhan14/

