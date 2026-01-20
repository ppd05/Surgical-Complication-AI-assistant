# ⚕️ Surgical Complication AI Assistant

> Intelligent RAG-powered clinical decision support system for surgical complication management using Gemini 2.0 Flash-Lite

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)](https://python.langchain.com/)

## 🌟 Overview

The **Surgical Complication AI Assistant** is an advanced clinical decision support tool that leverages Retrieval-Augmented Generation (RAG) architecture to provide evidence-based, structured information about surgical complications. Built with Gemini 2.0 Flash-Lite and LangChain, it delivers instant access to critical clinical protocols for diagnosis, risk assessment, and management of post-operative complications.

### 🎯 Key Features

- **🔍 Intelligent Search**: Instant protocol retrieval by surgery or complication name
- **📚 Knowledge Base Explorer**: Browse comprehensive surgical protocols via structured dropdowns
- **💬 AI-Powered Q&A**: Natural language queries with context-aware, structured responses
- **📋 Structured Clinical Output**: Organized by Diagnosis, Etiology, Risk Factors, and Management
- **⚡ Fast & Accurate**: Powered by Gemini 2.0 Flash-Lite for sub-2-second responses
- **🔐 Secure**: API keys managed via environment variables
- **🎨 User-Friendly UI**: Clean, intuitive Streamlit interface with tab-based navigation

## 🏗️ System Architecture
```
User Query
    ↓
Knowledge Base (JSON)
    ↓
Document Preparation & Chunking
    ↓
Vector Embeddings (HuggingFace: all-MiniLM-L6-v2)
    ↓
FAISS Vector Store
    ↓
Semantic Retrieval (Top 3 Similar Docs)
    ↓
RAG Chain (LangChain LCEL)
    ↓
Gemini 2.0 Flash-Lite (LLM)
    ↓
Structured Clinical Response
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Gemini 2.0 Flash-Lite | Fast inference, structured output generation |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Free local embeddings, 384-dimensional vectors |
| **Vector Store** | FAISS (CPU) | Efficient similarity search |
| **Framework** | LangChain | RAG pipeline orchestration |
| **UI** | Streamlit | Interactive web interface |
| **Language** | Python 3.8+ | Core development |

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get it free](https://makersuite.google.com/app/apikey))
- 2GB+ RAM (for embedding model)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ppd05/surgical-complication-assistant.git
cd surgical-complication-assistant
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*First installation takes 2-3 minutes (downloads embedding model)*

### 4. Configure API Key
```bash
# Copy example file
cp .env.example .env

# Edit .env and add your Gemini API key
# GOOGLE_API_KEY=your_actual_api_key_here
```

### 5. Run the Application
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📖 Usage Guide

### Mode 1: Instant Search (Fast)

1. Select **"💡 Instant Search (Fast)"** mode
2. Type any keyword (e.g., "anastomotic leak", "bile duct injury")
3. Instantly view the matching protocol

### Mode 2: Protocol Explorer (Dropdowns)

1. Select **"📚 Protocol Explorer (Dropdowns)"** mode
2. Choose surgery from first dropdown
3. Select specific complication from second dropdown
4. View detailed protocol

### Mode 3: AI Assistant Chat

1. Navigate to **"💬 AI Assistant Chat"** tab
2. Type your clinical question in natural language
3. Click **"Generate Structured Clinical Response"**
4. Receive formatted answer with sections:
   - Diagnosis & Presentation
   - Etiology & Risk Factors
   - Management Protocol
   - Source documents used

## 📊 Clinical Information Structure

Each protocol includes:

1. **Risk Factors**: Pre-existing conditions and procedural factors
2. **Etiology**: Mechanism of injury and pathophysiology
3. **Diagnostic Criteria**: Signs, symptoms, and diagnostic tests
4. **Management Protocol**: Step-by-step clinical management
5. **References**: Links to treatment guidelines

## 🗂️ Project Structure
```
surgical-complication-assistant/
│
├── src/                          # Core application modules
│   ├── __init__.py              # Package initializer
│   ├── ai_engine.py             # LLM and RAG chain setup
│   ├── data_loader.py           # Knowledge base loading
│   └── vectorstore_manager.py   # FAISS vector store management
│
├── data/                         # Knowledge base
│   └── surgical_knowledge.json  # Surgical protocols database
│
├── .streamlit/                   # Streamlit configuration
│   └── config.toml              # UI theme and settings
│
├── streamlit_app.py             # Main application entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Knowledge Base Format

The `surgical_knowledge.json` follows this structure:
```json
{
  "surgeries": [
    {
      "surgery_name": "Surgery Name",
      "category": "Surgical Category",
      "complications": [
        {
          "name": "Complication Name",
          "etiology": "Pathophysiology explanation",
          "risk_factors": "Risk factors list",
          "diagnostic_criteria": "Diagnostic signs and tests",
          "protocol": "Step 1: ... Step 2: ...",
          "references": ["URL1", "URL2"]
        }
      ]
    }
  ]
}
```

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in app settings:
```toml
GOOGLE_API_KEY = "your_api_key_here"
```

5. Deploy!

**Note:** Update the API key variable in `streamlit_app.py` from `GOOGLE_API_KEY` to match Streamlit secrets.

## 🔍 Technical Highlights

### RAG Pipeline

The system implements a sophisticated RAG pipeline:

1. **Document Preparation**: Concatenates all clinical fields (etiology, risk factors, diagnostics, protocol) for comprehensive context
2. **Chunking**: No chunking applied (documents are already atomic units)
3. **Embedding**: 384-dimensional vectors via sentence-transformers
4. **Retrieval**: Top-3 most similar documents using FAISS
5. **Generation**: Structured response with clinical sections

### LangChain LCEL

Uses LangChain Expression Language for clean, composable chains:
```python
rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

### Prompt Engineering

System prompt ensures:
- ✅ Responses grounded only in provided context
- ✅ Structured Markdown formatting
- ✅ Clinical section organization
- ✅ Source attribution

## 🎓 Use Cases

- **Medical Students**: Learning surgical complication management
- **Residents**: Quick reference during clinical rounds
- **Surgeons**: Evidence-based protocol refresher
- **Researchers**: Clinical decision support system prototype
- **Educators**: Teaching tool for surgical complications

## ⚠️ Disclaimer

**This application is for educational and research purposes only.**

- Not intended for real clinical use
- Not a substitute for professional medical judgment
- Not HIPAA compliant
- Always consult qualified healthcare professionals for medical decisions

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini** for the powerful LLM API
- **LangChain** for the RAG framework
- **HuggingFace** for free embedding models
- **Streamlit** for the excellent UI framework
- **FAISS** for efficient vector search
- **Ministry of Health & Family Welfare (MoHFW)** for treatment guidelines


## 🚫 PART 2: What NOT to Upload

**Delete or move these before uploading:**

1. ❌ `.env` file (contains your API key)
2. ❌ `venv/` or `env/` folder
3. ❌ `__pycache__/` folders
4. ❌ Any `.pyc` files
5. ❌ Any test data files

**Your final upload folder should contain ONLY:**
```
surgical-complication-assistant/
├── src/
│   ├── __init__.py
│   ├── ai_engine.py
│   ├── data_loader.py
│   └── vectorstore_manager.py
├── data/
│   └── surgical_knowledge.json
├── .streamlit/
│   └── config.toml
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .gitignore

└── README.md
