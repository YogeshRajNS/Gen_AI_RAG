# 📚 RAG Document Search & Summarization System
This README serves as both the technical documentation and detailed report 
as required by the interview assignment.

A production-ready **Retrieval-Augmented Generation (RAG)** system that enables intelligent document search and AI-powered summarization using Google Gemini AI, with comprehensive evaluation metrics.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Problem Statement

Design a system that can **search and summarize vast amounts of textual data efficiently** using Large Language Models (LLMs). The system should:

1. **Ingest and process** PDF documents
2. **Search** the corpus using hybrid retrieval methods
3. **Summarize** results with adjustable lengths
4. **Evaluate** quality using automated metrics (ROUGE, BLEU)
5. Provide a **user-friendly interface** for interaction

This project was developed as an interview assignment for a GenAI Engineer role.

## 🎯 Key Features

### Core Functionality
- 📄 **PDF Document Processing**: Automated ingestion, chunking, and indexing
- 🔍 **Hybrid Search Engine**: Combines vector embeddings + TF-IDF for superior accuracy
- ✨ **AI-Powered Summarization**: Google Gemini 2.0 Flash with adjustable summary lengths
- 🗄️ **Vector Database**: ChromaDB for efficient similarity search
- 🎨 **Modern Web Interface**: Interactive Streamlit dashboard

### Advanced Features
- 📊 **Quality Evaluation**: ROUGE and BLEU scores for summary quality
- 🎯 **Relevance Scoring**: Cosine similarity metrics for search accuracy
- 🧪 **Batch Testing**: Automated test suite with aggregated statistics
- 📈 **Real-time Metrics**: Live performance monitoring
- 🔄 **Session Management**: Persistent state across interactions

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (Streamlit)                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │   Upload   │ │   Search   │ │ Evaluation │ │  Testing │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
┌────────────────────────▼────────────────────────────────────┐
│                     BACKEND (Flask)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │             Search Engine                             │   │
│  │  ┌──────────────────┐  ┌─────────────────────────┐  │   │
│  │  │ Sentence Encoder │  │     TF-IDF Vectorizer   │  │   │
│  │  │  (MiniLM-L6-v2)  │  │      (Sci-kit Learn)    │  │   │
│  │  └──────────────────┘  └─────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────┐  ┌────────────────────────────────┐   │
│  │   ChromaDB       │  │    Google Gemini AI            │   │
│  │ Vector Database  │  │  Summarization Engine          │   │
│  └──────────────────┘  └────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Evaluation Framework                          │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ ROUGE Score │  │  BLEU Score  │  │  Cosine    │  │   │
│  │  │  (Summary)  │  │  (Summary)   │  │ Similarity │  │   │
│  │  └─────────────┘  └──────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## 🧮 Algorithm & Methodology

### 1. Document Processing Pipeline

```python
PDF Input → Text Extraction → Text Cleaning → Chunking (500 words, 50 overlap)
    ↓
Embedding Generation (Sentence Transformers)
    ↓
Vector Storage (ChromaDB) + TF-IDF Indexing
```

**Chunking Strategy:**
- **Size**: 500 words per chunk
- **Overlap**: 50 words between chunks
- **Rationale**: Maintains context while ensuring manageable chunk sizes

### 2. Hybrid Search Mechanism

The system combines two complementary search approaches:

#### A. Vector Similarity Search
- **Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Method**: Cosine similarity in embedding space
- **Strength**: Captures semantic meaning

#### B. TF-IDF Search
- **Method**: Traditional keyword-based retrieval
- **Strength**: Exact keyword matching

#### C. Hybrid Fusion
```python
Results = Vector_Search(query, top_k) ∪ TFIDF_Search(query, top_k)
Final = Remove_Duplicates(Results)[:top_k]
```

### 3. Summarization with Google Gemini

```python
Context = Concatenate(Retrieved_Chunks)
Prompt = f"Summarize in {length} words: {Context}"
Summary = Gemini_2.0_Flash.generate(Prompt)
```

**Adjustable Lengths:**
- **Short**: ~100 words
- **Medium**: ~200 words
- **Long**: ~400 words

### 4. Evaluation Metrics

#### Summary Quality Metrics

**ROUGE Scores (Recall-Oriented Understudy for Gisting Evaluation):**
- **ROUGE-1**: Unigram overlap (word-level matching)
- **ROUGE-2**: Bigram overlap (phrase-level matching)
- **ROUGE-L**: Longest common subsequence (sentence structure)

**BLEU Score (Bilingual Evaluation Understudy):**
- Measures n-gram precision with smoothing
- Commonly used for machine translation and generation tasks

#### Search Relevance Metrics

**Cosine Similarity:**
```python
similarity = (embedding_retrieved · embedding_expected) / 
             (||embedding_retrieved|| × ||embedding_expected||)
```

- **Max Similarity**: Best match score
- **Avg Similarity**: Average across all retrieved documents
- **Relevance Score**: Overall retrieval quality

### Complexity Analysis

- **Ingestion**: O(n × m) where n = pages, m = chunks per page
- **Search**: O(k × d) where k = top_k, d = embedding dimension
- **Summarization**: O(l) where l = context length
- **Space**: O(n × d) for vector storage

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Google Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))
- 4GB+ RAM recommended
- Internet connection for API calls

### Step 1: Clone the Repository

```bash
git clone https://github.com/YogeshRajNS/RAG_Document_Summarization.git
cd RAG_Document_Summarization
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Key

Create an `api.json` file in the root directory:

```json
{
  "api_key": "YOUR_GOOGLE_GEMINI_API_KEY_HERE"
}
```

**Alternative**: Set environment variable (for production)

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Then uncomment line 8 in `GenAI_rag.py`:
```python
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
```

### Step 5: Download NLTK Data (for BLEU scoring)

```python
python -c "import nltk; nltk.download('punkt')"
```

## 💻 Usage Guide

### Running the Application

#### 1. Start the Backend Server

```bash
python GenAI_rag.py
```

The Flask backend will start on `http://localhost:8000`

**Expected Output:**
```
 * Running on http://0.0.0.0:8000
 * Debug mode: off
```

#### 2. Launch the Frontend (New Terminal)

```bash
streamlit run frontend_for_rag.py
```

The Streamlit interface will open at `http://localhost:8501`

### Using the Interface

#### Tab 1: 📤 Upload & Query

1. **Upload Document**
   - Click "Browse files" and select a PDF
   - Click "🚀 Process Document"
   - Wait for confirmation

2. **Ask Questions**
   - Enter your query in the text box
   - Select summary length (Short/Medium/Long)
   - Click "🔍 Search & Summarize"

3. **View Results**
   - Generated summary appears in a green box
   - Source information shows number of chunks used

#### Tab 2: 🔍 Advanced Search

- **Search Without Summarization**: Get raw relevant chunks
- **Adjust Results**: Change `top_k` parameter (1-10)
- **View Metadata**: See document sources and page numbers

#### Tab 3: 📊 Evaluation

- **Test Summary Quality**: Compare against reference summaries
- **Evaluate Search**: Check retrieval accuracy
- **View Metrics**: ROUGE, BLEU, and similarity scores

#### Tab 4: 🧪 Batch Testing

- **Run Multiple Tests**: Test suite with JSON input
- **Aggregate Statistics**: Average performance metrics
- **Compare Results**: Side-by-side analysis

## 📊 API Endpoints

### Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "online",
  "message": "RAG Backend API is running",
  "endpoints": ["/upload", "/search", "/summarize", "/evaluate"]
}
```

### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

file: <PDF file>
```

**Response:**
```json
{
  "status": "success",
  "message": "Document uploaded and processed successfully",
  "filename": "document.pdf"
}
```

### Search Documents
```http
POST /search
Content-Type: application/json

{
  "query": "your search query",
  "top_k": 5
}
```

**Response:**
```json
{
  "status": "success",
  "query": "your search query",
  "results": ["chunk1", "chunk2", ...],
  "count": 5
}
```

### Summarize
```http
POST /summarize
Content-Type: application/json

{
  "query": "your query",
  "length": "medium"
}
```

**Response:**
```json
{
  "status": "success",
  "query": "your query",
  "summary": "Generated summary text...",
  "source_chunks": 5,
  "length": "medium"
}
```

### Evaluate
```http
POST /evaluate
Content-Type: application/json

{
  "query": "your query",
  "reference_summary": "expected summary",
  "reference_doc": "expected document content",
  "length": "medium"
}
```

**Response:**
```json
{
  "status": "success",
  "evaluation": {
    "generated_summary": "...",
    "retrieved_chunks": 5,
    "summary_metrics": {
      "rouge": { "rouge1": {...}, "rouge2": {...}, "rougeL": {...} },
      "bleu": { "bleu_score": 0.65 }
    },
    "search_metrics": {
      "similarities": [0.85, 0.78, ...],
      "max_similarity": 0.85,
      "avg_similarity": 0.76,
      "relevance_score": 0.85
    }
  }
}
```

### Batch Testing
```http
POST /test
Content-Type: application/json

{
  "test_cases": [
    {
      "query": "question 1",
      "expected_doc": "content",
      "reference_summary": "summary"
    }
  ]
}
```

## 🧪 Testing & Evaluation

### Manual Testing

1. Upload a test document (e.g., research paper)
2. Create queries with known answers
3. Evaluate summary quality visually
4. Check relevance of retrieved chunks

### Automated Testing

Use the Batch Testing tab with test cases:

```json
[
  {
    "query": "What is machine learning?",
    "reference_summary": "Machine learning is...",
    "expected_doc": "Machine learning content..."
  }
]
```

### Evaluation Criteria

**Summary Quality:**
- ✅ ROUGE-1 F1 > 0.5 (Good word overlap)
- ✅ ROUGE-L F1 > 0.4 (Good structure)
- ✅ BLEU Score > 0.3 (Adequate generation quality)

**Search Relevance:**
- ✅ Max Similarity > 0.7 (Strong match found)
- ✅ Avg Similarity > 0.5 (Overall relevant results)

## 📁 Project Structure

```
RAG_Document_Summarization/
│
├── GenAI_rag.py                    # Backend Flask API
├── frontend_for_rag.py             # Frontend Streamlit interface
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
├── Gen_AI_Engineer_-_RAG.pdf      # Problem statement
│
├── api.json                        # API configuration (create this)
│   └── { "api_key": "your_key" }
│
├── uploads/                        # Temporary PDF storage (auto-created)
├── chroma_store/                   # Vector database (auto-created)
│
└── .gitignore                      # Git ignore rules
```

## 🛠️ Technology Stack

### Backend Technologies
- **Flask 3.0**: Web framework for REST API
- **Flask-CORS**: Cross-origin resource sharing
- **PyMuPDF**: PDF text extraction
- **Sentence Transformers**: Text embedding (all-MiniLM-L6-v2)
- **ChromaDB**: Vector database for similarity search
- **scikit-learn**: TF-IDF vectorization
- **Google Gemini AI**: Text summarization
- **ROUGE Score**: Summary evaluation
- **NLTK**: BLEU score calculation
- **NumPy**: Numerical computations

### Frontend Technologies
- **Streamlit 1.29**: Interactive web interface
- **Requests**: HTTP client for API calls

### Key Models & Libraries

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embeddings | all-MiniLM-L6-v2 | 384-dim sentence embeddings |
| Vector DB | ChromaDB | Persistent vector storage |
| LLM | Google Gemini 2.0 Flash | Summarization |
| Search | TF-IDF + Cosine Similarity | Hybrid retrieval |
| Evaluation | ROUGE + BLEU | Quality metrics |

## 🎓 Methodology Details

### Data Preparation

1. **PDF Extraction**: PyMuPDF extracts text page by page
2. **Text Cleaning**: Remove extra whitespace, normalize
3. **Chunking**: Split into 500-word chunks with 50-word overlap
4. **Embedding**: Generate 384-dim vectors using Sentence Transformers
5. **Indexing**: Store in ChromaDB + build TF-IDF matrix

### Search Strategy

**Why Hybrid Search?**
- Vector search excels at semantic understanding
- TF-IDF excels at exact keyword matching
- Combining both provides robust retrieval

**Example:**
```
Query: "neural network architectures"

Vector Search finds:
- "deep learning models"
- "artificial neural nets"
- "convolutional networks"

TF-IDF finds:
- "neural network design"
- "network architecture patterns"

Combined: Best of both worlds!
```

### Evaluation Framework

The system implements comprehensive evaluation:

1. **Precision**: How many retrieved docs are relevant?
2. **Recall**: How many relevant docs were retrieved?
3. **F1-Score**: Harmonic mean of precision and recall
4. **Similarity**: Semantic closeness to expected content

## ⚙️ Configuration Options

### Backend Configuration (GenAI_rag.py)

```python
# Chunking parameters
CHUNK_SIZE = 500      # Words per chunk
CHUNK_OVERLAP = 50    # Overlap between chunks

# Search parameters
DEFAULT_TOP_K = 5     # Number of results to return

# Server settings
PORT = 8000           # Backend port
DEBUG = False         # Production mode
```

### Frontend Configuration (frontend_for_rag.py)

```python
# Backend URL
BACKEND = "http://localhost:8000"

# Summary lengths
LENGTHS = {
    "short": "100",
    "medium": "200",
    "long": "400"
}
```

## 🔒 Security & Best Practices

### API Key Management
- ✅ Store API keys in separate `api.json` file
- ✅ Add `api.json` to `.gitignore`
- ✅ Use environment variables in production
- ❌ Never commit API keys to version control

### Production Deployment
- Use `gunicorn` or `uvicorn` for Flask backend
- Enable HTTPS for secure communication
- Implement rate limiting for API endpoints
- Add authentication for sensitive operations
- Monitor API usage and costs

### Resource Management
- Clear upload directory periodically
- Implement document size limits
- Add timeout handling for API calls
- Monitor ChromaDB storage size

## 📈 Performance Optimization

### Speed Improvements
1. **Batch Processing**: Process multiple chunks simultaneously
2. **Caching**: Store frequently accessed embeddings
3. **Index Optimization**: Regular ChromaDB index maintenance
4. **Async Operations**: Use async/await for I/O operations

### Scalability Considerations
- Horizontal scaling with multiple Flask workers
- Distributed ChromaDB for large datasets
- Load balancing for high traffic
- Caching layer (Redis) for frequent queries

## 🐛 Troubleshooting

### Common Issues

**1. Backend won't start**
```bash
# Check if port 8000 is available
lsof -i :8000

# Kill existing process if needed
kill -9 <PID>
```

**2. "Backend Offline" in frontend**
```bash
# Ensure backend is running
ps aux | grep GenAI_rag

# Check backend health
curl http://localhost:8000/
```

**3. NLTK data missing**
```python
import nltk
nltk.download('punkt')
```

**4. ChromaDB errors**
```bash
# Remove and recreate ChromaDB
rm -rf chroma_store/
# Restart backend
```

**5. API key errors**
```bash
# Verify api.json exists and is valid
cat api.json
# Check JSON formatting
python -c "import json; print(json.load(open('api.json')))"
```

## 👤 Author

**Yogesh Raj NS** - [@YogeshRajNS](https://github.com/YogeshRajNS)

Project Link: [https://github.com/YogeshRajNS/RAG_Document_Summarization](https://github.com/YogeshRajNS/RAG_Document_Summarization)

## 🙏 Acknowledgments

- Problem based on **GenAI Engineer - RAG** interview assignment
- Built with Google Gemini AI for summarization
- Uses Sentence Transformers for embeddings
- ChromaDB for vector storage
- Evaluation framework with ROUGE and BLEU metrics
- Streamlit for modern web interface

## 🔮 Future Enhancements

-  Multi-document cross-referencing
-  Support for additional file formats (DOCX, TXT, HTML)
-  Advanced query understanding with query expansion
-  User authentication and document management
-  Export summaries to PDF/DOCX
-  Real-time collaborative document analysis
-  Integration with cloud storage (Google Drive, Dropbox)
-  Custom fine-tuning of embedding models
-  Conversation history and context retention
-  Advanced visualization dashboards

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: nsyogeshraj@gmail.com
- Documentation: See README sections above

---
