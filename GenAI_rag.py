import os
import json
import re
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

# ---------------- CONFIG ----------------

UPLOAD_DIR = "./uploads"
CHROMA_DIR = "./chroma_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Use environment variable for API key (for deployment)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# For local testing, uncomment below and comment above
# with open("api.json") as f:
#     api = json.load(f)
# genai.configure(api_key=api["api_key"])

# ---------------- UTILS ----------------

def clean_text(text):
    """Clean text by removing extra whitespace"""
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
    return chunks

# ---------------- EVALUATION METRICS ----------------

def calculate_rouge_scores(reference, generated):
    """
    Calculate ROUGE scores for summary evaluation
    ROUGE-1: Unigram overlap
    ROUGE-2: Bigram overlap
    ROUGE-L: Longest common subsequence
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'fmeasure': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'fmeasure': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'fmeasure': scores['rougeL'].fmeasure
            }
        }
    except ImportError:
        return {"error": "rouge-score package not installed. Run: pip install rouge-score"}

def calculate_bleu_score(reference, generated):
    """Calculate BLEU score for summary evaluation"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        reference_tokens = [reference.split()]
        generated_tokens = generated.split()
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing)
        return {"bleu_score": score}
    except ImportError:
        return {"error": "nltk package not installed. Run: pip install nltk"}

def evaluate_search_relevance(retrieved_docs, expected_doc_content):
    """
    Evaluate search relevance using cosine similarity
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    expected_emb = embedder.encode(expected_doc_content)
    retrieved_embs = embedder.encode(retrieved_docs)
    
    # Calculate cosine similarities
    similarities = []
    for ret_emb in retrieved_embs:
        similarity = np.dot(expected_emb, ret_emb) / (
            np.linalg.norm(expected_emb) * np.linalg.norm(ret_emb)
        )
        similarities.append(float(similarity))
    
    return {
        'similarities': similarities,
        'max_similarity': max(similarities) if similarities else 0,
        'avg_similarity': np.mean(similarities) if similarities else 0,
        'relevance_score': max(similarities) if similarities else 0
    }

# ---------------- ENGINE ----------------

class SearchEngine:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import chromadb

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="./chroma_store")

        self.collection = self.client.get_or_create_collection("docs")

        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_docs = []
        self.tfidf_matrix = None

    def ingest_pdf(self, path, name):
        """Ingest PDF, chunk it, and store in vector database"""
        import fitz
        pdf = fitz.open(path)
        texts = []

        for page in pdf:
            cleaned = clean_text(page.get_text())
            chunks = chunk_text(cleaned)

            for i, chunk in enumerate(chunks):
                emb = self.embedder.encode(chunk).tolist()

                self.collection.add(
                    documents=[chunk],
                    embeddings=[emb],
                    metadatas=[{"doc": name, "page": page.number + 1}],
                    ids=[f"{name}_{page.number}_{i}"]
                )
                texts.append(chunk)

        self.tfidf_docs.extend(texts)
        if self.tfidf_docs:
            self.tfidf_matrix = self.tfidf.fit_transform(self.tfidf_docs)

    def search(self, query, top_k=5):
        """
        Hybrid search using both vector embeddings and TF-IDF
        Returns top_k most relevant chunks
        """
        q_emb = self.embedder.encode(query).tolist()
        
        # Vector search
        vec = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k
        )

        results = vec["documents"][0] if vec["documents"] else []
        
        # TF-IDF search (if we have documents)
        if self.tfidf_matrix is not None and len(self.tfidf_docs) > 0:
            q_tfidf = self.tfidf.transform([query])
            tfidf_scores = np.dot(self.tfidf_matrix, q_tfidf.T).toarray().ravel()
            tfidf_idx = tfidf_scores.argsort()[-top_k:][::-1]
            
            # Add TF-IDF results
            results += [self.tfidf_docs[i] for i in tfidf_idx if i < len(self.tfidf_docs)]

        # Remove duplicates while preserving order
        return list(dict.fromkeys(results))[:top_k]

# engine = SearchEngine()
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = SearchEngine()
    return engine
# ---------------- GEMINI ----------------

def summarize(text, length):
    """Generate summary using Google Gemini"""
    size = {"short": "100", "medium": "200", "long": "400"}[length]

    prompt = f"""
Summarize the following content in approximately {size} words.
Use only the provided text. Be concise and capture the main points.

{text}
"""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    return model.generate_content(prompt).text

# ---------------- APP ----------------

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "RAG Backend API is running",
        "endpoints": ["/upload", "/search", "/summarize", "/evaluate"]
    })

@app.route("/upload", methods=["POST"])
def upload():
    """Upload and process PDF document"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(path)

        engine = get_engine()
        engine.ingest_pdf(path, file.filename)
        os.remove(path)

        return jsonify({
            "status": "success",
            "message": "Document uploaded and processed successfully",
            "filename": file.filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search():
    """Search for relevant documents"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Query parameter required"}), 400
        
        engine = get_engine()
        results = engine.search(data["query"], data.get("top_k", 5))
        return jsonify({
            "status": "success",
            "query": data["query"],
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summarize", methods=["POST"])
def summarize_api():
    """Search and summarize relevant documents"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Query parameter required"}), 400
        
        # Search for relevant documents
        engine = get_engine()
        docs = engine.search(data["query"])

        
        if not docs:
            return jsonify({
                "error": "No relevant documents found",
                "summary": "No documents available to summarize."
            }), 404
        
        combined = "\n\n".join(docs)
        summary = summarize(combined, data.get("length", "medium"))
        
        return jsonify({
            "status": "success",
            "query": data["query"],
            "summary": summary,
            "source_chunks": len(docs),
            "length": data.get("length", "medium")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Evaluate summary quality and search relevance
    
    Expected JSON body:
    {
        "query": "your query",
        "reference_summary": "expected summary",
        "reference_doc": "expected document content",
        "length": "medium"
    }
    """
    try:
        data = request.json
        
        if not data or 'query' not in data:
            return jsonify({"error": "Query parameter required"}), 400
        
        # Perform search
        engine = get_engine()
        retrieved_docs = engine.search(data["query"])

        
        # Generate summary
        combined = "\n\n".join(retrieved_docs)
        generated_summary = summarize(combined, data.get("length", "medium"))
        
        evaluation_results = {
            "query": data["query"],
            "generated_summary": generated_summary,
            "retrieved_chunks": len(retrieved_docs)
        }
        
        # Evaluate summary quality if reference provided
        if 'reference_summary' in data and data['reference_summary']:
            rouge_scores = calculate_rouge_scores(
                data['reference_summary'], 
                generated_summary
            )
            bleu_score = calculate_bleu_score(
                data['reference_summary'],
                generated_summary
            )
            evaluation_results['summary_metrics'] = {
                'rouge': rouge_scores,
                'bleu': bleu_score
            }
        
        # Evaluate search relevance if reference doc provided
        if 'reference_doc' in data and data['reference_doc']:
            relevance_scores = evaluate_search_relevance(
                retrieved_docs,
                data['reference_doc']
            )
            evaluation_results['search_metrics'] = relevance_scores
        
        return jsonify({
            "status": "success",
            "evaluation": evaluation_results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["POST"])
def run_test_suite():
    """
    Run automated test suite with predefined queries
    
    Expected JSON body:
    {
        "test_cases": [
            {
                "query": "machine learning",
                "expected_doc": "content about ML",
                "reference_summary": "ML is..."
            }
        ]
    }
    """
    try:
        data = request.json
        
        if not data or 'test_cases' not in data:
            return jsonify({"error": "test_cases parameter required"}), 400
        
        results = []
        
        for test_case in data['test_cases']:
            query = test_case.get('query')
            expected_doc = test_case.get('expected_doc', '')
            reference_summary = test_case.get('reference_summary', '')
            
            # Search
            engine = get_engine()
            retrieved_docs = engine.search(query)

            
            # Summarize
            combined = "\n\n".join(retrieved_docs)
            generated_summary = summarize(combined, "medium")
            
            test_result = {
                "query": query,
                "retrieved_chunks": len(retrieved_docs)
            }
            
            # Calculate metrics
            if reference_summary:
                rouge = calculate_rouge_scores(reference_summary, generated_summary)
                test_result['rouge_scores'] = rouge
            
            if expected_doc:
                relevance = evaluate_search_relevance(retrieved_docs, expected_doc)
                test_result['relevance_score'] = relevance['relevance_score']
            
            results.append(test_result)
        
        # Calculate average scores
        avg_rouge1 = np.mean([r.get('rouge_scores', {}).get('rouge1', {}).get('fmeasure', 0) 
                               for r in results if 'rouge_scores' in r])
        avg_relevance = np.mean([r.get('relevance_score', 0) for r in results])
        
        return jsonify({
            "status": "success",
            "test_results": results,
            "summary_statistics": {
                "total_tests": len(results),
                "avg_rouge1_f1": float(avg_rouge1),
                "avg_relevance_score": float(avg_relevance)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    app.run(host="0.0.0.0", port=port, debug=False)


