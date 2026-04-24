# 📚 StudyMate

> An AI-powered study companion that helps you learn smarter — not harder.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gemini API](https://img.shields.io/badge/Google-Gemini%20Flash%20API-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![IBM watsonx](https://img.shields.io/badge/IBM-watsonx-054ADA?logo=ibm&logoColor=white)](https://www.ibm.com/watsonx)
[![FAISS](https://img.shields.io/badge/Meta-FAISS-0064AF?logo=meta&logoColor=white)](https://faiss.ai/)

---

## 🧠 Overview

**StudyMate** is an AI-powered Q&A and study assistant application built using the **Google Gemini Flash API**, **IBM watsonx**, and **FAISS** (Facebook AI Similarity Search). It empowers students, teachers, and curious minds to extract knowledge from documents, generate flashcards and quizzes, and receive real-time study guidance — all within a clean, intuitive interface.

Developed by **Team TripleByte** as part of a Hackathon project.

---

## ✨ Features

- 📄 **Document Upload & Analysis** — Upload PDF or document files and instantly query their content
- 🔍 **Intelligent Information Retrieval** — Powered by FAISS for fast, semantic search across uploaded material
- 🃏 **Flashcard Generation** — Auto-generate flashcards from study material for active recall
- 📝 **Quiz Generation** — Create quizzes to test understanding of uploaded content
- 💡 **Real-Time Study Tips** — Personalized study guidance based on user context
- 📒 **Notes & Work Saving** — Save progress and notes within the app
- 📱 **Responsive UI** — Minimalist ocean-blue themed interface built with Streamlit

---

## 🎯 Target Users

| User Type | Use Case |
|---|---|
| 🎓 **Students** | Better access to information and self-paced learning |
| 👩‍🏫 **Teachers** | Explain concepts in an engaging and efficient manner |
| 🔭 **Intellectual Explorers** | Discover and understand new concepts with ease |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, HTML, CSS |
| **Backend / AI** | Google Gemini Flash API, IBM watsonx |
| **Vector Search** | FAISS |
| **Language** | Python 3.10+ |
| **Database** | Not required (API-based queries) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A valid **Google Gemini API Key**
- An **IBM watsonx** account and credentials
- `pip` package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/studymate.git
   cd studymate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the root directory:

   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   IBM_WATSONX_API_KEY=your_ibm_watsonx_api_key
   IBM_WATSONX_PROJECT_ID=your_project_id
   IBM_WATSONX_URL=your_watsonx_url
   ```

4. **Run the application**

   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`

---

## 📖 Usage

1. **Upload** a PDF or document file using the upload panel
2. **Ask questions** or choose to generate flashcards/quizzes from the uploaded content
3. The app retrieves semantically relevant content via **FAISS** and calls the **Gemini Flash API**
4. Results are displayed in a clean, easy-to-read format
5. **Save notes** or export generated flashcards for later review

---

## 🏗️ System Architecture

```
User (Streamlit UI)
       │
       ▼
  Document Upload
       │
       ▼
 FAISS Vector Store ──► Similarity Search
       │
       ▼
 Gemini Flash API + IBM watsonx
       │
       ▼
  Processed Results
  (Q&A / Flashcards / Quizzes / Tips)
       │
       ▼
  Frontend Display
```

---

## 🧪 Testing Summary

| Test Case | Category | Status |
|---|---|---|
| Frontend UI — file upload, flashcard & quiz generation | Functional | ✅ Passed |
| Backend API — information retrieval via Gemini | Functional | ✅ Passed |
| API response time under 500ms | Performance | ⚠️ Needs Optimization |
| Incorrect API response fixes | Bug Fix | ✅ Fixed |
| Responsive UI across mobile & desktop | Validation | 🔧 In Progress |

---

## 🗓️ Development Roadmap

- [x] Environment setup & API integration
- [x] Basic frontend UI with input fields
- [x] FAISS-based document search
- [x] Flashcard and quiz generation
- [x] Error handling & API stability
- [ ] Mobile-responsive UI
- [ ] API response caching for performance
- [ ] User authentication & profile management
- [ ] Export flashcards to PDF/Anki format

---

## 🤝 Team

**Team TripleByte**

| Name | Role |
|---|---|
| Jahnavi Priya P | Frontend Development & Bug Fixes |
| Lahari M | API Integration & Backend |
| Manmitha M | Research, Testing & UI Enhancements |

---

## ⚠️ Known Issues

- **API Response Time** — Some queries may exceed 500ms; caching optimizations are in progress
- **Mobile UI** — Layout issues on smaller screens are currently being resolved

---

## 📄 License

This project was developed for hackathon purposes. All rights reserved by Team TripleByte.

---

<p align="center">Made with ❤️ by Team TripleByte</p>
