import streamlit as st
import requests
import json

# Get backend URL from secrets (or fallback to localhost for local dev)
#BACKEND = st.secrets.get("BACKEND_URL", "http://localhost:8000")
BACKEND = "http://localhost:8000"
# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .query-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .summary-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin-top: 1rem;
    }
    .metric-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin-top: 0.5rem;
    }
    .evaluation-box {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin-top: 1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .stSelectbox>div>div>select {
        border-radius: 8px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=RAG+AI", width=150)
    st.title("📚 RAG Assistant")
    st.markdown("---")
    st.markdown("""
    ### About
    This application uses **Retrieval-Augmented Generation (RAG)** to:
    - 📄 Process PDF documents
    - 🔍 Search through content
    - ✨ Generate AI summaries
    - 📊 Evaluate quality metrics
    
    ### Technology Stack
    - 🤖 **Google Gemini 2.0 Flash**
    - 🧠 **Sentence Transformers**
    - 🗄️ **ChromaDB Vector Store**
    - 📊 **TF-IDF Search**
    - 🎯 **ROUGE & BLEU Evaluation**
    """)
    st.markdown("---")
    
    # Backend status check
    try:
        response = requests.get(f"{BACKEND}/", timeout=5)
        if response.status_code == 200:
            st.success("🟢 Backend Online")
        else:
            st.error("🔴 Backend Error")
    except:
        st.error("🔴 Backend Offline")
    
    st.markdown("---")
    st.info("💡 Upload a PDF and ask questions to get AI-powered insights!")
    
    # Show uploaded documents
    if st.session_state.uploaded_docs:
        st.markdown("### 📁 Uploaded Documents")
        for doc in st.session_state.uploaded_docs:
            st.text(f"✓ {doc}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🤖 Document Search & Summarization")
    st.markdown("### Powered by Google Gemini AI")

with col2:
    st.metric("Status", "🟢 Online", "Ready")

st.markdown("---")

# Create tabs for different features
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Query", "🔍 Advanced Search", "📊 Evaluation", "🧪 Batch Testing"])

# TAB 1: Upload & Query (Main functionality)
with tab1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    col_upload1, col_upload2 = st.columns([3, 1])
    
    with col_upload1:
        st.subheader("📤 Upload Your Document")
        st.markdown("Upload a PDF document to add it to the knowledge base")
        file = st.file_uploader("", type=["pdf"], label_visibility="collapsed", key="upload_tab1")
    
    with col_upload2:
        st.write("")
        st.write("")
        if file:
            st.success(f"✅ {file.name}")
            if st.button("🚀 Process Document", key="process_tab1"):
                with st.spinner("🔄 Processing document..."):
                    try:
                        r = requests.post(
                            f"{BACKEND}/upload",
                            files={"file": file},
                            timeout=60
                        )
                        if r.status_code == 200:
                            st.success("✅ Document processed successfully!")
                            st.session_state.uploaded_docs.append(file.name)
                            st.balloons()
                        else:
                            st.error(f"❌ Upload failed: {r.json().get('error', 'Unknown error')}")
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Query Section
    st.markdown('<div class="query-section">', unsafe_allow_html=True)
    
    st.subheader("🔍 Ask Questions & Get Summaries")
    
    col_query1, col_query2 = st.columns([3, 1])
    
    with col_query1:
        query = st.text_input(
            "Enter your question",
            placeholder="e.g., What are the main findings in the document?",
            label_visibility="visible",
            key="query_tab1"
        )
    
    with col_query2:
        length = st.selectbox(
            "Summary Length",
            ["short", "medium", "long"],
            index=1,
            help="Choose how detailed you want the summary to be",
            key="length_tab1"
        )
    
    if st.button("✨ Generate Summary", use_container_width=True, key="summarize_tab1"):
        if not query:
            st.warning("⚠️ Please enter a question first!")
        else:
            with st.spinner("🤔 Searching documents and generating summary..."):
                try:
                    res = requests.post(
                        f"{BACKEND}/summarize",
                        json={"query": query, "length": length},
                        timeout=60
                    )
                    
                    if res.status_code == 200:
                        result = res.json()
                        summary = result["summary"]
                        st.session_state.last_summary = summary
                        
                        st.markdown("---")
                        st.subheader("📋 Summary")
                        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                        
                        # Additional info
                        col_info1, col_info2, col_info3 = st.columns(3)
                        with col_info1:
                            st.metric("Query", "✅ Completed")
                        with col_info2:
                            st.metric("Length", length.capitalize())
                        with col_info3:
                            st.metric("Words", len(summary.split()))
                        
                        # Source chunks info
                        if 'source_chunks' in result:
                            st.info(f"📚 Generated from {result['source_chunks']} relevant document chunks")
                            
                    else:
                        error_msg = res.json().get('error', 'Unknown error')
                        st.error(f"❌ Failed to generate summary: {error_msg}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Advanced Search
with tab2:
    st.subheader("🔍 Advanced Document Search")
    st.markdown("Search for specific content without summarization")
    
    search_query = st.text_input(
        "Search Query",
        placeholder="Enter keywords or phrases to search",
        key="search_query_tab2"
    )
    
    top_k = st.slider("Number of results", 1, 10, 5, key="top_k_tab2")
    
    if st.button("🔎 Search Documents", key="search_tab2"):
        if not search_query:
            st.warning("⚠️ Please enter a search query!")
        else:
            with st.spinner("🔍 Searching..."):
                try:
                    res = requests.post(
                        f"{BACKEND}/search",
                        json={"query": search_query, "top_k": top_k},
                        timeout=30
                    )
                    
                    if res.status_code == 200:
                        result = res.json()
                        results = result.get("results", [])
                        
                        st.success(f"✅ Found {len(results)} relevant chunks")
                        
                        for i, chunk in enumerate(results, 1):
                            with st.expander(f"📄 Result {i}"):
                                st.markdown(chunk)
                    else:
                        st.error(f"❌ Search failed: {res.json().get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# TAB 3: Evaluation
with tab3:
    st.subheader("📊 Summary & Search Evaluation")
    st.markdown("Evaluate the quality of search results and summaries using ROUGE, BLEU, and relevance metrics")
    
    eval_query = st.text_input(
        "Query",
        placeholder="Enter your evaluation query",
        key="eval_query_tab3"
    )
    
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        eval_length = st.selectbox(
            "Summary Length",
            ["short", "medium", "long"],
            index=1,
            key="eval_length_tab3"
        )
    
    with col_eval2:
        st.write("")
    
    # Optional reference inputs
    st.markdown("#### Optional: Provide References for Comparison")
    
    reference_summary = st.text_area(
        "Reference Summary (optional)",
        placeholder="Paste expected summary here for ROUGE/BLEU comparison",
        height=100,
        key="ref_summary_tab3"
    )
    
    reference_doc = st.text_area(
        "Reference Document Content (optional)",
        placeholder="Paste expected document content for relevance evaluation",
        height=100,
        key="ref_doc_tab3"
    )
    
    if st.button("📊 Evaluate Query", use_container_width=True, key="evaluate_tab3"):
        if not eval_query:
            st.warning("⚠️ Please enter a query!")
        else:
            with st.spinner("🔬 Evaluating..."):
                try:
                    eval_data = {
                        "query": eval_query,
                        "length": eval_length
                    }
                    
                    if reference_summary:
                        eval_data["reference_summary"] = reference_summary
                    if reference_doc:
                        eval_data["reference_doc"] = reference_doc
                    
                    res = requests.post(
                        f"{BACKEND}/evaluate",
                        json=eval_data,
                        timeout=60
                    )
                    
                    if res.status_code == 200:
                        result = res.json()
                        evaluation = result["evaluation"]
                        st.session_state.evaluation_results = evaluation
                        
                        # Display generated summary
                        st.markdown("---")
                        st.subheader("📋 Generated Summary")
                        st.markdown(f'<div class="summary-box">{evaluation["generated_summary"]}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Display metrics if available
                        if 'summary_metrics' in evaluation:
                            st.markdown("---")
                            st.subheader("📈 Summary Quality Metrics")
                            
                            metrics = evaluation['summary_metrics']
                            
                            # ROUGE Scores
                            if 'rouge' in metrics and 'error' not in metrics['rouge']:
                                st.markdown("#### ROUGE Scores")
                                rouge = metrics['rouge']
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                    st.metric("ROUGE-1 F1", f"{rouge['rouge1']['fmeasure']:.4f}")
                                    st.caption("Word overlap")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                    st.metric("ROUGE-2 F1", f"{rouge['rouge2']['fmeasure']:.4f}")
                                    st.caption("Phrase overlap")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                    st.metric("ROUGE-L F1", f"{rouge['rougeL']['fmeasure']:.4f}")
                                    st.caption("Sentence structure")
                                    st.markdown('</div>', unsafe_allow_html=True)
                            
                            # BLEU Score
                            if 'bleu' in metrics and 'error' not in metrics['bleu']:
                                st.markdown("#### BLEU Score")
                                bleu = metrics['bleu']
                                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                st.metric("BLEU Score", f"{bleu.get('bleu_score', 0):.4f}")
                                st.caption("Translation/generation quality")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Search relevance metrics
                        if 'search_metrics' in evaluation:
                            st.markdown("---")
                            st.subheader("🎯 Search Relevance Metrics")
                            
                            search = evaluation['search_metrics']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                st.metric("Max Similarity", f"{search['max_similarity']:.4f}")
                                st.caption("Best match score")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                st.metric("Avg Similarity", f"{search['avg_similarity']:.4f}")
                                st.caption("Average relevance")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                st.metric("Relevance Score", f"{search['relevance_score']:.4f}")
                                st.caption("Overall relevance")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Overall info
                        st.info(f"📚 Evaluated using {evaluation['retrieved_chunks']} document chunks")
                        
                    else:
                        st.error(f"❌ Evaluation failed: {res.json().get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# TAB 4: Batch Testing
with tab4:
    st.subheader("🧪 Batch Testing")
    st.markdown("Run multiple test cases at once and get aggregated statistics")
    
    # Example test cases
    default_test_cases = """[
  {
    "query": "What is machine learning?",
    "reference_summary": "Machine learning is a method of data analysis that automates analytical model building using algorithms that learn from data.",
    "expected_doc": "Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from data."
  },
  {
    "query": "What are neural networks?",
    "reference_summary": "Neural networks are computing systems inspired by biological neural networks that learn to perform tasks by considering examples.",
    "expected_doc": "Neural networks consist of interconnected nodes that process information using a connectionist approach to computation."
  }
]"""
    
    st.markdown("#### Test Cases (JSON format)")
    test_cases_input = st.text_area(
        "Enter test cases",
        value=default_test_cases,
        height=300,
        key="test_cases_tab4"
    )
    
    if st.button("🧪 Run Batch Tests", use_container_width=True, key="batch_test_tab4"):
        try:
            test_cases = json.loads(test_cases_input)
            
            with st.spinner("🧪 Running tests..."):
                res = requests.post(
                    f"{BACKEND}/test",
                    json={"test_cases": test_cases},
                    timeout=120
                )
                
                if res.status_code == 200:
                    result = res.json()
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("📊 Summary Statistics")
                    
                    stats = result['summary_statistics']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tests", stats['total_tests'])
                    with col2:
                        st.metric("Avg ROUGE-1 F1", f"{stats['avg_rouge1_f1']:.4f}")
                    with col3:
                        st.metric("Avg Relevance", f"{stats['avg_relevance_score']:.4f}")
                    
                    # Individual results
                    st.markdown("---")
                    st.subheader("📋 Individual Test Results")
                    
                    for i, test_result in enumerate(result['test_results'], 1):
                        with st.expander(f"Test {i}: {test_result['query']}", expanded=False):
                            col_test1, col_test2 = st.columns(2)
                            
                            with col_test1:
                                st.markdown("**Retrieved Chunks:**")
                                st.write(test_result['retrieved_chunks'])
                                
                                if 'rouge_scores' in test_result:
                                    st.markdown("**ROUGE Scores:**")
                                    rouge = test_result['rouge_scores']
                                    st.write(f"- ROUGE-1: {rouge['rouge1']['fmeasure']:.4f}")
                                    st.write(f"- ROUGE-2: {rouge['rouge2']['fmeasure']:.4f}")
                                    st.write(f"- ROUGE-L: {rouge['rougeL']['fmeasure']:.4f}")
                            
                            with col_test2:
                                if 'relevance_score' in test_result:
                                    st.markdown("**Relevance Score:**")
                                    st.write(f"{test_result['relevance_score']:.4f}")
                    
                    st.success("✅ All tests completed successfully!")
                    
                else:
                    st.error(f"❌ Batch test failed: {res.json().get('error', 'Unknown error')}")
                    
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON format. Please check your test cases.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with ❤️ using Streamlit, Flask, and Google Gemini AI</p>
        <p style='font-size: 0.8rem;'>© 2024 RAG Document Assistant | Complete with Evaluation Framework</p>
    </div>
""", unsafe_allow_html=True)