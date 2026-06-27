# RAG for AGORA AI Governance Research Assistant (Ollama Phi-3.5-Mini, HF MPNet-V2)

## Output example:

<img width="640" height="360" alt="image" src="https://github.com/user-attachments/assets/95b0f631-9863-4d27-8072-41bd998575e1" />

## ⚛︎ Project overview:
This project utilize RAG systems to develop research assistance for AGORA AI Governance Documents, sourced from https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data. 

The system is powered by:
- 🤗 HuggingFace Mpnet-b2 model for text embeddings
- 🦙 Ollama Phi-3.5-Mini generation model for inference
- 🛢 Chroma for vector database.

See contents:
- [System Overview](#system-overview)
- [Installation](#-installation)

## System Overview
### 💠 Components:
- **Embedding model**: Embed feature text semantics and user query into vector embeddings for vector search.
- **Vector database**: Store text data: vector, text, metadata. Perform vector search (Metadata search is not impemented in this project).
- **Generation model**: Generate response from given prompt (augmented prompt).

### 🔀 Flow:
1. User prompt query.
2. Query is passed to vector database search, through embedding model and perform search (similarity/max marginal relevenace).
3. Vector database return retrieved Documents.
4. Augment generation model prompts through adding context from Documents, plus system instructions.
5. User received response from generation model.

### ⚙️ Development:
- **Preprocessing**:
   - Transform each modular semantic entities into contextful Document.
   - Context injection: Adding document's overview and usecase semantic as high-level context into text content.
   - Chunking: Chunk large texts into smaller chunks, with chunk overlaps.
   - Metadata injection: Adding semantic text metadata include unique identifier (AGORA Document ID), categories (Authorities, usecases), and chunk portions. 
- **Injection**:
  - Utilize Chroma DB as vector database for its vector search, embeded embedding function, and file-based database capability.
  - Stream documents using Python's lazy iterator 'Generator' for efficient memory use.
  - Add all documents entities with embedding models into vector database.
- **Prompt Engineering**:
  - Provide Generation Model systems instructions to maximize context understanding, accurrate response, minimize hallucinations.
  - Provide Generation Model professional tonality.
  - Augment Generation Model user prompts to always include text high-level context and metadata from retrievals. 
- **Application**:
  - Modularly designed classes for ochestrating usecase applications.
  - CLI-based interface
  - Stream-based Generation Model response (Generator, yield) to combat frozen, inefficient memory use.
     
## 💻 Installation
1. **Clone the repo**
```bash
git clone https://github.com
cd RAG-Agora
```
2. **Setup environment**

```bash
# Initialize LFS to pull required large size files
git lfs install
git lfs pull

# Install required dependencies
pip install -r requirements.txt
```

3. **Run application**
```bash
python -m app.main
```

