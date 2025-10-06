# Fix for OpenMP library conflict on macOS - MUST BE FIRST
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.chat_models import ChatOpenAI
import pickle
from pathlib import Path
import hashlib
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import queue

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Create directories
VECTOR_STORE_DIR = Path("vector_stores")
PDF_WATCH_DIR = Path("pdfs_to_process")
PDF_PROCESSED_WORK_DIR = Path("pdfs_processed")
VECTOR_STORE_DIR.mkdir(exist_ok=True)
PDF_WATCH_DIR.mkdir(exist_ok=True)
PDF_PROCESSED_WORK_DIR.mkdir(exist_ok=True)


# ✅ Initialize session state FIRST - before any other code
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

# Global queue for processing PDFs in background
processing_queue = queue.Queue()

# ✅ Thread-safe lock for session state access
processing_lock = threading.Lock()


def hash_file_content(file_path=None, file_bytes=None):
    """Create consistent hash from file content"""
    if file_bytes:
        return hashlib.md5(file_bytes).hexdigest()
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def update_processing_status(file_name, status):
    """Thread-safe way to update processing status"""
    with processing_lock:
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
        st.session_state.processing_status[file_name] = status


def process_pdf_background(file_path, file_name):
    """Process PDF in background thread"""
    try:
        print(f"🔄 Starting PDF processing for: {file_name}")
        update_processing_status(file_name, "processing")

        file_path = Path(file_path)
        file_hash = hash_file_content(file_path=file_path)
        print(f"📊 Generated file hash: {file_hash} for {file_name}")

        vector_store_path = VECTOR_STORE_DIR / f"{file_hash}"
        metadata_path = VECTOR_STORE_DIR / f"{file_hash}_metadata.pkl"

        # Check if already processed
        if vector_store_path.exists():
            print(f"💾 Found cached vector store for {file_name}")
            update_processing_status(file_name, "cached")
            return file_hash, "cached"

        print(f"📖 Reading PDF content from: {file_path}")
        # Process the PDF
        pdf_reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            text += page.extract_text()
            if page_num % 10 == 0:  # Log every 10 pages
                print(f"📄 Processed {page_num + 1} pages for {file_name}")
        
        print(f"📝 Extracted {len(text)} characters from {len(pdf_reader.pages)} pages in {file_name}")

        # Break into chunks
        print(f"✂️ Splitting text into chunks for {file_name}")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        print(f"📦 Created {len(chunks)} chunks for {file_name}")

        # Create embeddings
        print(f"🧠 Creating embeddings for {file_name}")
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)
        print(f"✅ Created vector store with {len(chunks)} embeddings for {file_name}")

        # Save to disk
        print(f"💾 Saving vector store to: {vector_store_path}")
        vector_store.save_local(str(vector_store_path))

        metadata = {
            'total_chunks': len(chunks),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        print(f"📋 Saving metadata to: {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"🎉 Successfully completed processing {file_name} - {len(chunks)} chunks created")
        update_processing_status(file_name, "completed")
        return file_hash, "completed", len(chunks)

    except Exception as e:
        print(f"❌ Error processing {file_name}: {str(e)}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        update_processing_status(file_name, f"error: {str(e)}")
        return None, "error"


def load_vector_store(file_hash):
    """Load vector store from disk"""
    vector_store_path = VECTOR_STORE_DIR / f"{file_hash}"
    metadata_path = VECTOR_STORE_DIR / f"{file_hash}_metadata.pkl"

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.load_local(
        str(vector_store_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    return vector_store, metadata


def get_available_pdfs():
    """Get list of all processed PDFs"""
    vector_stores = list(VECTOR_STORE_DIR.glob("*_metadata.pkl"))
    pdf_info = []

    for metadata_file in vector_stores:
        try:
            with open(metadata_file, 'rb') as f:
                meta = pickle.load(f)
                file_hash = metadata_file.stem.replace('_metadata', '')
                pdf_info.append({
                    'name': meta['file_name'],
                    'hash': file_hash,
                    'chunks': meta['total_chunks'],
                    'processed_at': meta.get('processed_at', 'Unknown')
                })
        except:
            continue

    return pdf_info


# Watchdog Handler
class PDFHandler(FileSystemEventHandler):
    """Monitors PDF folder and queues files for processing"""

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.pdf'):
            return

        # Add to processing queue instead of blocking
        file_name = Path(event.src_path).name
        
        # ✅ Thread-safe check if file is already being processed
        with processing_lock:
            if 'processing_status' not in st.session_state:
                st.session_state.processing_status = {}
            
            if file_name not in st.session_state.processing_status:
                processing_queue.put((event.src_path, file_name))
                st.session_state.processing_status[file_name] = "queued"
    
    def on_modified(self, event):
        # ✅ Also handle file modifications (sometimes upload triggers this instead of created)
        if event.is_directory or not event.src_path.endswith('.pdf'):
            return
            
        file_name = Path(event.src_path).name
        
        # ✅ Thread-safe check - only process if not already in queue/processing
        with processing_lock:
            if 'processing_status' not in st.session_state:
                st.session_state.processing_status = {}
                
            if file_name not in st.session_state.processing_status:
                processing_queue.put((event.src_path, file_name))
                st.session_state.processing_status[file_name] = "queued"


# Background processor thread
def background_processor():
    """Process PDFs from queue in background"""
    print("🚀 Background processor started!")  # Debug log
    while True:
        try:
            file_path, file_name = processing_queue.get(timeout=1)
            print(f"📥 Processing: {file_name} from {file_path}")  # Debug log
            
            # ✅ Wait for file to be fully written and verify it exists
            time.sleep(1)  # Reduced wait time
            
            if not Path(file_path).exists():
                print(f"❌ File not found: {file_path}")  # Debug log
                update_processing_status(file_name, "error: File not found")
                continue
                
            # ✅ Verify file is readable and has content
            try:
                file_size = Path(file_path).stat().st_size
                if file_size == 0:
                    print(f"❌ File is empty: {file_path}")  # Debug log
                    update_processing_status(file_name, "error: File is empty")
                    continue
                    
                with open(file_path, 'rb') as test_file:
                    test_content = test_file.read(1024)  # Try to read first 1KB
                    if not test_content:
                        print(f"❌ File has no content: {file_path}")  # Debug log
                        update_processing_status(file_name, "error: File has no content")
                        continue
                        
            except Exception as e:
                print(f"❌ File not readable: {file_path} - {str(e)}")  # Debug log
                update_processing_status(file_name, f"error: File not readable - {str(e)}")
                continue

            try:
                result = process_pdf_background(file_path, file_name)
                print(f"✅ Completed processing: {file_name} - {result}")  # Debug log
            except Exception as processing_error:
                print(f"❌ Error in process_pdf_background for {file_name}: {str(processing_error)}")
                import traceback
                print(f"🔍 Processing traceback: {traceback.format_exc()}")
                update_processing_status(file_name, f"error: {str(processing_error)}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"💥 Background processing error: {e}")
            import traceback
            traceback.print_exc()


# Initialize watchdog observer
@st.cache_resource
def start_watchdog():
    """Start the file system observer"""
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, str(PDF_WATCH_DIR), recursive=False)
    observer.start()

    # Start background processor thread
    processor_thread = threading.Thread(target=background_processor, daemon=True)
    processor_thread.start()

    return observer


## Streamlit UI
st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide")
st.header("📚 PDF Chatbot with Background Processing")

# Start watchdog
observer = start_watchdog()

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")

    # Processing status
    if st.session_state.processing_status:
        st.subheader("⏳ Processing Status")
        for file_name, status in st.session_state.processing_status.items():
            if status == "queued":
                st.info(f"🔄 {file_name}: Queued")
            elif status == "processing":
                st.warning(f"⚙️ {file_name}: Processing...")
            elif status == "completed":
                st.success(f"✅ {file_name}: Completed")
            elif status == "cached":
                st.info(f"💾 {file_name}: Loaded from cache")
            elif status.startswith("error"):
                st.error(f"❌ {file_name}: {status}")

    st.divider()

    # Show monitoring status
    st.success(f"👀 Monitoring: `{PDF_WATCH_DIR}/`")

    # Manual upload option
    st.subheader("📤 Manual Upload")
    file = st.file_uploader("Upload a PDF file", type="pdf", key="manual_upload")

    if file is not None:
        # Save to watch directory for processing
        temp_path = PDF_WATCH_DIR / file.name

        if not temp_path.exists() or st.button("Re-process"):
            with open(temp_path, 'wb') as f:
                f.write(file.read())
            
            # ✅ Force manual processing trigger
            file_name = file.name
            processing_queue.put((str(temp_path), file_name))
            update_processing_status(file_name, "queued")
            
            st.success(f"📄 {file.name} added to processing queue!")
            st.rerun()
        
        # ✅ Add manual process button for existing files
        elif temp_path.exists():
            if st.button(f"🔄 Process {file.name}", key=f"process_{file.name}"):
                file_name = file.name
                processing_queue.put((str(temp_path), file_name))
                update_processing_status(file_name, "queued")
                st.success(f"📄 {file.name} re-added to processing queue!")
                st.rerun()
        
        # ✅ Add debug info for uploaded file
        st.info(f"File exists: {temp_path.exists()}")
        if temp_path.exists():
            st.info(f"File size: {temp_path.stat().st_size} bytes")

    st.divider()

    # PDF Library
    st.subheader("📂 PDF Library")
    available_pdfs = get_available_pdfs()

    if available_pdfs:
        st.success(f"💾 {len(available_pdfs)} PDF(s) available")

        # Create selection options
        pdf_options = [f"{pdf['name']} ({pdf['chunks']} chunks)" for pdf in available_pdfs]

        # selected_idx = st.selectbox(
        #     "Select PDF to query:",
        #     range(len(pdf_options)),
        #     format_func=lambda x: pdf_options[x],
        #     key="pdf_selector"
        # )
        ## multiple pf selections
        selected_indices = st.multiselect(
            "Select one or more PDFs to query:",
            range(len(pdf_options)),
            format_func=lambda x: pdf_options[x],
            key="pdf_selector_multi"
        )

        # if st.button("📖 Load Selected PDF", use_container_width=True):
        #     selected_pdf = available_pdfs[selected_idx]
        #     try:
        #         vector_store, metadata = load_vector_store(selected_pdf['hash'])
        #         st.session_state.vector_store = vector_store
        #         st.session_state.metadata = metadata
        #         st.session_state.current_pdf = selected_pdf['name']
        #         st.success(f"✅ Loaded: {selected_pdf['name']}")
        #         st.rerun()
        #     except Exception as e:
        #         st.error(f"Error loading PDF: {e}")

        if st.button("📖 Load Selected PDFs", use_container_width=True):
            if not selected_indices:
                st.warning("Please select at least one PDF.")
            else:
                try:
                    all_vectors = []
                    all_metadata = []
                    for idx in selected_indices:
                        selected_pdf = available_pdfs[idx]
                        vector_store, metadata = load_vector_store(selected_pdf['hash'])
                        all_vectors.append(vector_store)
                        all_metadata.append(metadata)

                    # ✅ Merge FAISS vector stores together
                    combined_store = all_vectors[0]
                    for v in all_vectors[1:]:
                        combined_store.merge_from(v)

                    # ✅ Store in session for Q&A
                    st.session_state.vector_store = combined_store
                    st.session_state.metadata = {
                        'merged_from': [pdf['file_name'] for pdf in all_metadata]
                    }
                    st.session_state.current_pdf = ", ".join(
                        [pdf['file_name'] for pdf in all_metadata]
                    )

                    st.success(
                        f"✅ Loaded {len(selected_indices)} PDFs: {st.session_state.current_pdf}"
                    )
                    st.rerun()

                except Exception as e:
                    st.error(f"Error loading PDFs: {e}")

        # Show details of available PDFs
        with st.expander("📋 View All PDFs"):
            for pdf in available_pdfs:
                st.text(f"• {pdf['name']}")
                st.caption(f"  Chunks: {pdf['chunks']} | Processed: {pdf['processed_at']}")
    else:
        st.warning("No PDFs processed yet")

    st.divider()

    # Clear cache option
    if st.button("🗑️ Clear All Cache", use_container_width=True):
        for file in VECTOR_STORE_DIR.glob("*"):
            file.unlink()
        st.session_state.clear()
        st.success("Cache cleared!")
        st.rerun()

    # Instructions
    with st.expander("📖 How to use"):
        st.markdown(f"""
        **Auto-Processing (Non-blocking):**
        1. Drop PDF files into `{PDF_WATCH_DIR}/`
        2. Files are queued and processed in background
        3. Continue asking questions on current PDF
        4. Switch to new PDF when ready

        **Manual Upload:**
        1. Use file uploader above
        2. File added to processing queue
        3. No waiting required!

        **Ask Questions:**
        - Select a processed PDF from library
        - Type your question below
        - Get AI-powered answers instantly

        **Key Feature:**
        ✨ Upload new PDFs without interrupting your current chat session!
        """)

    # Add this in the sidebar after the processing status section
    if st.checkbox("🐛 Debug Mode"):
        st.subheader("🔍 Debug Info")
        st.write(f"Queue size: {processing_queue.qsize()}")
        st.write(f"Watch directory: {PDF_WATCH_DIR}")
        st.write(f"Files in watch dir: {list(PDF_WATCH_DIR.glob('*.pdf'))}")
        st.write(f"Observer running: {observer.is_alive()}")
        
        # ✅ Add manual trigger for stuck files
        st.subheader("🔧 Manual Processing")
        pdf_files = list(PDF_WATCH_DIR.glob('*.pdf'))
        if pdf_files:
            for pdf_file in pdf_files:
                file_name = pdf_file.name
                current_status = st.session_state.processing_status.get(file_name, "unknown")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.text(f"{file_name}: {current_status}")
                with col2:
                    if st.button("🚀 Force Process", key=f"force_{file_name}"):
                        processing_queue.put((str(pdf_file), file_name))
                        update_processing_status(file_name, "queued")
                        st.success(f"Forced processing of {file_name}")
                        st.rerun()
        else:
            st.info("No PDF files in watch directory")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Display current PDF info
    if "current_pdf" in st.session_state:
        if 'merged_from' in st.session_state.metadata:
            st.info(f"📄 **Current PDFs:** {st.session_state.current_pdf}")
        else:
            st.info(f"📄 **Current PDF:** {st.session_state.current_pdf} | "
                    f"**Chunks:** {st.session_state.metadata['total_chunks']}")
    else:
        st.info("👈 Select a PDF from the sidebar to get started!")

with col2:
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh", value=False,
                               help="Automatically refresh to show processing updates")

    if auto_refresh:
        st.caption("Refreshing every 3 seconds...")
        time.sleep(3)
        st.rerun()

# Question answering section
if "vector_store" in st.session_state:
    user_question = st.text_input(
        "💬 Ask a question about the PDF:",
        key="user_question_input",
        placeholder="Type your question here..."
    )

    if user_question:
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                # Search (LOCAL - no blocking)
                match = st.session_state.vector_store.similarity_search(user_question, k=4)

                # Show chunks (optional)
                with st.expander("📄 View retrieved chunks"):
                    for i, doc in enumerate(match, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content[:300] + "...")
                        st.divider()

                # Generate answer
                llm = ChatOpenAI(
                    api_key=OPENAI_API_KEY,
                    temperature=0,
                    max_tokens=1000,
                    model="gpt-3.5-turbo"
                )

                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=match, question=user_question)

                # Display answer
                st.subheader("💡 Answer:")
                st.write(response)

            except Exception as e:
                st.error(f"Error generating answer: {e}")
else:
    st.markdown("""
    ### 🚀 Getting Started

    1. **Upload or drop PDFs** into the `pdfs_to_process/` folder
    2. **Processing happens in background** - no need to wait!
    3. **Select a PDF** from the sidebar library
    4. **Ask questions** and get instant answers

    **Pro Tip:** You can upload new PDFs while chatting with current ones! 🎉
    """)

# Footer with status
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"🟢 Watchdog Active")
with col2:
    processing_count = sum(1 for s in st.session_state.processing_status.values()
                           if s in ["queued", "processing"])
    if processing_count > 0:
        st.caption(f"⚙️ {processing_count} PDF(s) processing")
    else:
        st.caption("✅ No active processing")
with col3:
    st.caption(f"💾 {len(get_available_pdfs())} PDF(s) in library")