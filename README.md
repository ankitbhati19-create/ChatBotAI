# 📚 AI-Powered Prodcut oferring Discovery and Plan Discovery chatBot

A Streamlit-based chatbot that allows you to upload PDF documents and ask questions about their content. Features **non-blocking background processing**, automatic file monitoring with Watchdog, and persistent vector store caching. Powered by OpenAI's GPT-3.5-turbo and FAISS vector search for intelligent, context-aware responses.

## ✨ Key Features

### Core Functionality
- 📄 **PDF Document Processing** - Upload and extract text from PDF files
- 🔍 **Semantic Search** - Find relevant information using FAISS vector similarity
- 💬 **Natural Language Answers** - Get human-like responses powered by GPT-3.5-turbo
- 💾 **Persistent Storage** - Vector embeddings cached locally (MD5 hash-based)
- ⚡ **Instant Loading** - Previously processed PDFs load in <1 second

### Advanced Features
- 🚀 **Non-Blocking Processing** - Upload PDFs without interrupting your chat session
- 🔄 **Background Queue System** - Multiple PDFs processed sequentially in background thread
- 👀 **Auto-Detection with Watchdog** - Automatically detect and process PDFs dropped into folder
- 📚 **PDF Library Management** - Switch between multiple processed PDFs via dropdown
- 📊 **Real-Time Status Dashboard** - Live processing status for all PDFs
- 🔒 **Secure** - API keys stored in environment variables
- 💰 **Cost Optimized** - Embeddings created once, cached forever

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one at [platform.openai.com](https://platform.openai.com))

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <your-project-folder>
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

4. **Run the application**

**Recommended: Integrated Non-Blocking Mode**
```bash
 streamlit run /Users/abhati/PycharmProjects/ChatBotNew/chatbotwithBackGroundProcessing.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Dependencies

Create a `requirements.txt` file with:

```
faiss-cpu==1.12.0
langchain==0.3.27
langchain-community==0.3.30
langchain-core==0.3.78
langchain-openai==0.3.34
langchain-text-splitters==0.3.11
numpy==2.3.3
openai==2.1.0
PyPDF2==3.0.1
python-dotenv==1.1.1
streamlit==1.50.0
tiktoken==0.11.0
watchdog==6.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

## 🎯 How It Works

### Architecture Overview

```
PDF Upload/Drop → Background Queue → Text Extraction → Text Chunking
                                                              ↓
                                                    Embeddings (OpenAI API)
                                                              ↓
                                            FAISS Vector Store (Saved to Disk)
                                                              ↓
User Question → Vector Search (Local) → Relevant Chunks → GPT-3.5 API
                                                              ↓
                                                   Natural Language Answer
```
<img width="2443" height="1244" alt="image" src="https://github.com/user-attachments/assets/8029f48b-eb9b-4eb4-9664-90eea2dde07a" />

### Non-Blocking Process Flow

#### PDF Processing (Background Thread)
1. **File Detection** - Watchdog detects new PDF in `pdfs_to_process/`
2. **Queue Management** - PDF added to processing queue
3. **Status Update** - UI shows "Queued" → "Processing" → "Completed"
4. **Background Processing** - Happens in separate thread
   - PDF text extraction (PyPDF2)
   - Text chunking (1000 chars, 150 overlap)
   - Embedding creation (OpenAI API call)
   - FAISS vector store creation
   - Persistent storage with MD5 hash
5. **Availability** - PDF appears in library dropdown when complete

#### Question Answering (Main Thread)
1. **User selects PDF** - Load from library dropdown
2. **User asks question** - Type in text input
3. **Local vector search** - Find relevant chunks (no API call)
4. **Context retrieval** - Get top 4 matching chunks
5. **GPT-3.5 generation** - Send chunks + question to OpenAI
6. **Answer display** - Show natural language response

### Key Technical Features

#### 1. Non-Blocking Design
- **Background thread** processes PDFs using Python threading
- **Queue system** manages multiple uploads sequentially
- **Main thread** stays responsive for user interactions
- **No spinners** blocking the UI during processing

#### 2. MD5 Hash-Based Caching
```python
# Consistent hashing ensures same file = same cache
file_hash = hashlib.md5(file_bytes).hexdigest()
# Example: "a1b2c3d4e5f6..." (always same for identical content)
```

Benefits:
- Same file uploaded twice = instant load
- File content changes = new cache created
- Persistent across app restarts
- No Python `hash()` randomness issues

#### 3. Multi-PDF Library
- All processed PDFs stored in `vector_stores/`
- Dropdown selector for instant switching
- Metadata includes: chunks, size, processing time
- No re-processing when switching

## 📁 Project Structure

```
your-project/
│
├── chatbotwithBackGroundProcessing.py              # Main Streamlit app (non-blocking, integrated watchdog)      # with background processor (optional)
├── .env                    # Environment variables (NOT committed)
├── .gitignore             # Git ignore rules
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License (optional)
│
├── pdfs_to_process/       # 👀 WATCHED: Drop PDFs here for auto-processing
│   └── (new PDFs detected automatically)
│
├── pdfs_processed/        # ✅ ARCHIVE: Completed PDFs moved here (optional)
│   └── (processed PDFs archived)
│
└── vector_stores/         # 💾 CACHE: Persistent embeddings storage
    ├── <md5_hash>.faiss       # FAISS index
    ├── <md5_hash>.pkl         # FAISS vectors
    └── <md5_hash>_metadata.pkl # File metadata
```

### Directory Details

| Directory | Purpose | Auto-Created | Can Delete |
|-----------|---------|--------------|------------|
| `pdfs_to_process/` | Watchdog monitors this folder | ✅ Yes | ⚠️ Clears queue |
| `pdfs_processed/` | Archive of completed PDFs | ✅ Yes | ✅ Safe to delete |
| `vector_stores/` | Cached embeddings | ✅ Yes | ⚠️ Loses cache |

## 🔑 Environment Variables

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | ✅ Yes | `sk-proj-abc123...` |

### Setting Up .env

```bash
# Create .env file
echo "OPENAI_API_KEY=sk-proj-your-actual-key-here" > .env

# Verify it's in .gitignore
echo ".env" >> .gitignore
```

## 💰 Cost Estimation

### API Usage Breakdown

| Operation | Cost | Frequency | Notes |
|-----------|------|-----------|-------|
| **Embeddings** | ~$0.0001/1K tokens | One-time per PDF | Cached forever |
| **GPT-3.5 Query** | ~$0.002/query | Per question | Input + output tokens |

### Real-World Examples

**Example 1: Small Document (10 pages)**
- Processing: ~$0.01 (one-time)
- 50 questions: ~$0.10
- **Total: $0.11** for complete usage

**Example 2: Large Document (100 pages)**
- Processing: ~$0.08 (one-time)
- 200 questions: ~$0.40
- **Total: $0.48** for complete usage

**Example 3: Multiple Documents (5 PDFs, 50 pages each)**
- Processing all: ~$0.25 (one-time)
- 100 questions across PDFs: ~$0.20
- **Total: $0.45** for complete usage

### Cost Optimization Features

✅ **MD5 hash caching** - Never re-embed same content  
✅ **Persistent storage** - Cache survives app restarts  
✅ **Efficient chunking** - Only relevant chunks sent to GPT  
✅ **Local vector search** - No API cost for finding chunks  
✅ **Background processing** - Process once, use forever  

### Monthly Cost Estimates

| Usage Level | PDFs/Month | Questions/Month | Estimated Cost |
|-------------|------------|-----------------|----------------|
| Light | 10 | 100 | ~$0.50 |
| Medium | 50 | 500 | ~$2.50 |
| Heavy | 200 | 2000 | ~$10.00 |

## 🎮 Usage Guide

### Method 1: Auto-Processing (Non-Blocking) ⭐ Recommended

**The magic of non-blocking processing:**

1. **Start the app**
   ```bash
   streamlit run chatbot.py
   ```

2. **Upload or drop PDFs** (both work!)
   ```bash
   # Option A: Drop via file system
   cp ~/Downloads/document.pdf pdfs_to_process/
   
   # Option B: Use manual upload in sidebar
   # (File uploader widget)
   ```

3. **Continue working immediately!**
   - PDF shows as "Queued" → "Processing"
   - You can still chat with current PDF
   - No waiting, no blocking!

4. **Switch when ready**
   - Check processing status in sidebar
   - When "✅ Completed", select from dropdown
   - Instant switching between PDFs

**Real-world scenario:**
```
9:00 AM - Upload research_paper.pdf
9:00 AM - Start asking questions immediately
9:02 AM - Upload annual_report.pdf (processing in background)
9:03 AM - Still chatting with research_paper.pdf
9:05 AM - annual_report.pdf ready, switch to it
9:06 AM - Ask questions about annual_report.pdf
```

### Method 2: Manual Upload

1. Click **"📤 Manual Upload"** in sidebar
2. Select PDF from your computer
3. File added to processing queue
4. Continue with current PDF while it processes

### Method 3: Batch Upload

Drop multiple PDFs at once:
```bash
# Copy multiple files
cp ~/Documents/*.pdf pdfs_to_process/

# Or drag-and-drop multiple files into folder
```

All files queued and processed sequentially in background!

### Method 4: Background Processor (Production)

For heavy workloads, run processor separately:

```bash
# Terminal 1: Dedicated processor
python pdf_processor.py
# Shows: real-time processing logs
# Benefits: 100% dedicated to processing

# Terminal 2: Streamlit UI
streamlit run chatbot.py
# Benefits: UI stays ultra-responsive
```

## 🎨 User Interface Guide

### Sidebar Components

#### 1. ⏳ Processing Status
- Real-time status for each PDF
- Icons: 🔄 Queued | ⚙️ Processing | ✅ Completed | ❌ Error

#### 2. 📤 Manual Upload
- File picker for direct upload
- Supports single PDF at a time
- Automatically added to processing queue

#### 3. 📂 PDF Library
- Dropdown of all processed PDFs
- Shows: Name (chunks count)
- Click "Load" to switch instantly
- Expandable list with details

#### 4. 🗑️ Clear All Cache
- Removes all processed PDFs
- Clears vector stores
- Fresh start

### Main Content Area

#### Current PDF Display
```
📄 Current PDF: research_paper.pdf | Chunks: 87
```

#### Question Input
```
💬 Ask a question about the PDF:
[Type your question here...]
```

#### Answer Display
```
💡 Answer:
[GPT-3.5 generated response appears here]

📄 View retrieved chunks [Expandable]
```

### Footer Status Bar

```
🟢 Watchdog Active | ⚙️ 2 PDF(s) processing | 💾 5 PDF(s) in library
```

### Auto-Refresh Toggle

- **Checkbox**: 🔄 Auto-refresh
- **When enabled**: Refreshes every 3 seconds
- **Purpose**: See live processing updates
- **Tip**: Disable when not uploading to save resources

## 🛠️ Configuration

### Adjust Chunk Size

Modify in `chatbot.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=1000,      # ⬆️ Increase for more context per chunk
    chunk_overlap=150,    # ⬆️ Increase for better continuity
    length_function=len
)
```

**Recommendations:**
- **Small chunks (500-800)**: Precise answers, less context
- **Medium chunks (1000-1500)**: Balanced (default)
- **Large chunks (2000-3000)**: More context, may dilute relevance

### Change Number of Retrieved Chunks

```python
match = vector_store.similarity_search(user_question, k=4)
```

- `k=2`: Fast, less context
- `k=4`: Balanced (default)
- `k=6-8`: More context, slower, higher cost

### Switch GPT Model

```python
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    temperature=0,           # 0 = consistent, 1 = creative
    max_tokens=1000,         # Max response length
    model="gpt-3.5-turbo"   # Options below
)
```

**Model Options:**
- `gpt-3.5-turbo`: Fast, cheap, good quality ✅ (default)
- `gpt-4`: Best quality, slower, 10x cost
- `gpt-4-turbo`: Faster GPT-4, still expensive

### Configure Watchdog Folders

```python
# In chatbot.py
VECTOR_STORE_DIR = Path("vector_stores")
PDF_WATCH_DIR = Path("pdfs_to_process")

# Change to custom paths
VECTOR_STORE_DIR = Path("/custom/cache/path")
PDF_WATCH_DIR = Path("/custom/upload/path")
```

### Adjust Background Thread Settings

```python
# In background_processor() function
processing_queue.get(timeout=1)  # Check queue every 1 second
time.sleep(1)  # Wait 1 second after file detected (ensure fully written)
```

### Auto-Refresh Interval

```python
# In main UI section
if auto_refresh:
    st.caption("Refreshing every 3 seconds...")
    time.sleep(3)  # Change to desired interval
    st.rerun()
```

## 🔍 Usage Tips & Best Practices

### PDF Requirements

✅ **Good PDFs:**
- Text-based PDFs (not scanned images)
- Clear, readable text
- Well-formatted documents
- English language (or configure for other languages)

❌ **Problematic PDFs:**
- Scanned images without OCR
- Password-protected files
- Corrupted or damaged files
- PDFs with complex layouts (may extract poorly)

### Question Best Practices

**Effective Questions:**
```
✅ "What are the main findings in chapter 3?"
✅ "Summarize the methodology section"
✅ "List all recommendations from the report"
✅ "What date was the contract signed?"
✅ "Compare the Q1 and Q2 results"
```

**Less Effective Questions:**
```
❌ "Tell me everything" (too broad)
❌ "What's your opinion?" (GPT only uses PDF content)
❌ "What's the weather?" (not in document)
❌ "Translate this" (not designed for translation)
```

### Workflow Optimization

**Morning Batch Processing:**
```bash
# 8:00 AM - Drop all PDFs for the day
cp ~/Documents/meetings/*.pdf pdfs_to_process/

# 8:05 AM - All processed, ready to use
# Work with any PDF instantly via dropdown
```

**Continuous Work:**
```
1. Start with PDF A
2. Upload PDF B (background processing)
3. Continue chatting with PDF A
4. Switch to PDF B when ready
5. Upload PDF C (background processing)
6. Continue pattern...
```

### Performance Tips

1. **Use auto-refresh sparingly** - Disable when not uploading
2. **Close unused PDFs** - Clear cache periodically
3. **Batch similar questions** - More efficient than one-by-one
4. **Pre-process overnight** - Drop PDFs before leaving
5. **Monitor costs** - Check OpenAI usage dashboard regularly

## 🐛 Troubleshooting

### Common Issues

#### OpenMP Library Conflict (macOS)
```
Error: OMP: Error #15: Initializing libomp.dylib...
```

**Solution:** Already handled in code
```python
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

If still occurs:
```bash
# Reinstall numpy
pip uninstall numpy
pip install numpy
```

#### Watchdog Not Detecting Files

**Symptoms:** PDFs dropped but not processing

**Solutions:**
1. Check folder exists: `ls pdfs_to_process/`
2. Verify file extension: Must be `.pdf` (lowercase)
3. Check terminal for errors
4. Restart the app
5. Verify watchdog running: Check sidebar status

**Debug:**
```python
# Add logging in PDFHandler.on_created()
print(f"File detected: {event.src_path}")
```

#### Processing Stuck

**Symptoms:** Status shows "Processing..." forever

**Solutions:**
1. Check OpenAI API key is valid
2. Check internet connection
3. Look for errors in terminal
4. Try re-uploading PDF
5. Clear cache and retry

**Force stop and restart:**
```bash
# Ctrl+C in terminal
# Delete problematic file from pdfs_to_process/
# Restart app
```

#### PDF Not in Dropdown

**Symptoms:** Processed but not appearing in library

**Solutions:**
1. Check `vector_stores/` folder exists
2. Verify metadata files: `ls vector_stores/*metadata.pkl`
3. Click refresh or restart app
4. Check for errors in sidebar status
5. Try enabling auto-refresh

**Manual check:**
```bash
ls -la vector_stores/
# Should see: hash.faiss, hash.pkl, hash_metadata.pkl
```

#### Memory Issues

**Symptoms:** App crashes with large PDFs

**Solutions:**
1. Process PDFs sequentially (already default)
2. Reduce chunk size in configuration
3. Increase system memory
4. Use background processor separately
5. Process large PDFs overnight

#### API Rate Limits

**Symptoms:** "Rate limit exceeded" errors

**Solutions:**
1. OpenAI free tier has limits
2. Upgrade to paid tier
3. Add delays between processing
4. Process during off-peak hours

### Error Messages Decoded

| Error | Meaning | Solution |
|-------|---------|----------|
| `No module named 'watchdog'` | Missing package | `pip install watchdog` |
| `API key not found` | .env not loaded | Check .env file exists |
| `File not found` | Wrong path | Verify folder structure |
| `Rate limit exceeded` | Too many API calls | Wait or upgrade plan |
| `PDF encrypted` | Password-protected | Remove password first |

## 🔒 Security Best Practices

### Critical Security Rules

⚠️ **NEVER commit `.env` to Git**  
⚠️ **NEVER share your OpenAI API key**  
⚠️ **NEVER hardcode API keys in code**  
⚠️ **Rotate keys if accidentally exposed**  

### Secure Setup

1. **Create .env properly**
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
chmod 600 .env  # Make it readable only by you
```

2. **Verify .gitignore**
```bash
cat .gitignore | grep .env
# Should output: .env
```

3. **Test before committing**
```bash
git status
# .env should NOT appear in untracked files
```

### .gitignore Template

```
# Environment variables
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environment
venv/
env/
ENV/
.venv

# Vector stores (optional - can be large)
vector_stores/

# Processed PDFs (optional)
pdfs_processed/
pdfs_to_process/*.pdf

# Streamlit
.streamlit/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Jupyter
.ipynb_checkpoints/

# Logs
*.log
```

### If API Key Exposed

1. **Immediately revoke** at https://platform.openai.com/api-keys
2. **Generate new key**
3. **Update .env file**
4. **Check OpenAI usage** for unauthorized charges
5. **Reset Git history** if key was committed:
```bash
# Use BFG Repo-Cleaner or git filter-branch
# Or create new repo if easier
```

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Free)

1. **Prepare repository**
```bash
# Ensure .env is in .gitignore
git add .gitignore
git commit -m "Add gitignore"
git push
```

2. **Deploy to Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect GitHub repository
- Select `chatbot.py` as main file

3. **Add secrets**
- In Streamlit Cloud dashboard
- Go to "Secrets" section
- Add:
```toml
OPENAI_API_KEY = "sk-proj-your-key-here"
```

4. **Deploy**
- Click "Deploy"
- Wait 2-3 minutes
- App is live!

**Note:** Watchdog may have limited functionality on Streamlit Cloud. Manual upload will work perfectly.

### Option 2: Local Network Access

Share with devices on your network:

```bash
# Find your IP address
# macOS/Linux
ifconfig | grep "inet "

# Windows
ipconfig

# Run with network access
streamlit run chatbot.py --server.address 0.0.0.0

# Access from other devices
http://<your-ip>:8501
```

### Option 3: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p pdfs_to_process pdfs_processed vector_stores

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
CMD ["streamlit", "run", "chatbot.py", "--server.address", "0.0.0.0"]
```

Build and run:
```bash
# Build image
docker build -t pdf-chatbot .

# Run container
docker run -p 8501:8501 --env-file .env pdf-chatbot

# Access at http://localhost:8501
```

### Option 4: VPS Deployment (DigitalOcean, AWS, etc.)

```bash
# SSH into server
ssh user@your-server-ip

# Clone repository
git clone <your-repo>
cd <your-repo>

# Install dependencies
pip install -r requirements.txt

# Set up .env
nano .env
# Add: OPENAI_API_KEY=your-key

# Run with tmux (keeps running after logout)
tmux new -s chatbot
streamlit run chatbot.py --server.port 8501 --server.address 0.0.0.0

# Detach: Ctrl+B then D
# Reattach: tmux attach -t chatbot
```

## 📊 Advanced Features

### Batch Processing Script

Create `batch_process.py`:

```python
from pathlib import Path
import time

# Drop all PDFs from a folder
source_folder = Path("~/Documents/to_process").expanduser()
dest_folder = Path("pdfs_to_process")

for pdf in source_folder.glob("*.pdf"):
    print(f"Copying {pdf.name}...")
    shutil.copy(pdf, dest_folder / pdf.name)
    time.sleep(2)  # Stagger uploads

print("All PDFs queued!")
```

### Export Processed PDF List

```python
import pickle
from pathlib import Path

def list_pdfs():
    for metadata_file in Path("vector_stores").glob("*_metadata.pkl"):
        with open(metadata_file, 'rb') as f:
            meta = pickle.load(f)
            print(f"{meta['file_name']}")
            print(f"  Chunks: {meta['total_chunks']}")
            print(f"  Processed: {meta['processed_at']}")
            print()

list_pdfs()
```

### Monitor Processing Queue

```python
# Add to chatbot.py to see queue size
if not processing_queue.empty():
    st.sidebar.info(f"📋 Queue: {processing_queue.qsize()} PDF(s)")
```

### Custom Event Handlers

Extend watchdog functionality:

```python
class CustomPDFHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # Handle PDF updates
        if event.src_path.endswith('.pdf'):
            print(f"PDF modified: {event.src_path}")
            # Re-process if needed
    
    def on_deleted(self, event):
        # Clean up associated vector stores
        if event.src_path.endswith('.pdf'):
            print(f"PDF deleted: {event.src_path}")
            # Remove from cache
```

## 🎓 Learning Resources

### Understanding the Technology

- **FAISS**: [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- **LangChain**: [Documentation](https://python.langchain.com/)
- **OpenAI API**: [API Reference](https://platform.openai.com/docs)
- **Streamlit**: [Getting Started](https://docs.streamlit.io/)
- **Watchdog**: [Documentation](https://pythonhosted.org/watchdog/)

### Recommended Reading

1. **RAG (Retrieval Augmented Generation)**
   - How vector search improves LLM responses
   - Chunking strategies
   - Embedding models comparison

2. **Prompt Engineering**
   - Effective context construction
   - Question decomposition
   - Response optimization

3. **Cost Optimization**
   - Token management
   - Caching strategies
   - API rate limiting

## 📈 Roadmap & Future Features

### Planned Features

- [x] PDF text extraction
- [x] Vector search with FAISS
- [x] Persistent caching with MD5 hashing
- [x] Watchdog auto-processing
- [x] Non-blocking background processing
- [x] Multi-PDF library management
- [x] Real-time processing status
- [ ] **OCR support** for scanned documents
- [ ] **Word document support** (.docx, .doc)
- [ ] **Chat history** persistence across sessions
- [ ] **Export conversations** to PDF/Markdown
- [ ] **Multi-language** support
- [ ] **Voice input/output** integration
- [ ] **Advanced analytics** dashboard
- [ ] **Collaborative features** (share PDFs with team)
- [ ] **API endpoint** for programmatic access
- [ ] **Mobile app** version

### Community Requests

Vote on features: [GitHub Issues](link-to-your-repo/issues)

## 🤝 Contributing

We welcome contributions! Here's how:

### Getting Started

1. **Fork the repository**
2. **Clone your fork**
```bash
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot
```

3. **Create a branch**
```bash
git checkout -b feature/amazing-feature
```

4. **Make changes**
- Write clean, documented code
- Follow existing code style
- Add comments where needed

5. **Test thoroughly**
```bash
# Test with various PDFs
# Test error cases
# Test with multiple concurrent uploads
```

6. **Commit**
```bash
git add .
git commit -m "Add amazing feature: description"
```

7. **Push and create PR**
```bash
git push origin feature/amazing-feature
# Create Pull Request on GitHub
```

### Contribution Guidelines

- ✅ Write clear commit messages
- ✅ Update README if needed
- ✅ Add comments to complex code
- ✅ Test edge cases
- ✅ Follow Python PEP 8 style
- ✅ Update requirements.txt if adding dependencies

## 📝 License

This project is licensed under 

```
MIT License

Copyright (c) 2024 [Ankit Bhati]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

Built with amazing open-source technologies:

- **[Streamlit](https://streamlit.io)** - Beautiful web apps for ML/data science
- **[OpenAI](https://openai.com)** - GPT-3.5-turbo language model
- **[FAISS](https://github.com/facebookresearch/faiss)** - Efficient similarity search
- **[LangChain](https://langchain.com)** - LLM application framework
- **[Watchdog](https://github.com/gorakhargosh/watchdog)** - File system monitoring
- **[PyPDF2](https://pypdf2.readthedocs.io/)** - PDF text extraction

## 📧 Support & Contact

### Getting Help

1. **Check this README** - Most questions answered here
2. **Search existing issues** - Someone may have solved it
3. **Open a new issue** - Describe your problem clearly
4. **Community forum** - [Link to discussion board]

### Reporting Bugs

Include:
- Python version
- Operating system
- Full error message
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests

Open an issue with:
- Clear description of feature
- Use case / why it's valuable
- Proposed implementation (optional)

## 🌟 Star This Project

If you find this useful, please give it a ⭐ on GitHub!

### Share It

- Tweet about it: `#PDFChatbot #AI #OpenAI`
- Write a blog post
