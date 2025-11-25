# Testing Guide for RAG Implementation

## Quick Start

### 1. Activate Virtual Environment

```bash
cd /mnt/c/Users/AndreyPopov/ai-dial-rag-overview
source .venv/bin/activate  # On Windows WSL
```

### 2. Run Interactive Test Script (For Screenshots)

```bash
python Test.py
```

This will run through 6 test cases:
- **Test 1-4**: Valid queries about the microwave manual (should provide accurate answers)
- **Test 5-6**: Invalid queries (should indicate inability to answer from context)

Press Enter after each test to proceed to the next one. This allows you to take screenshots of each result.

### 3. Run Interactive RAG Assistant

```bash
python task/app.py
```

This launches the interactive RAG assistant where you can ask your own questions about the microwave manual.

## Test Cases Included

### Valid Queries (Expected to Answer)
1. "What safety precautions should be taken to avoid exposure to excessive microwave energy?"
2. "What is the maximum cooking time that can be set on the DW 395 HCG microwave oven?"
3. "How should you clean the glass tray of the microwave oven?"
4. "What materials are safe to use in this microwave during both microwave and grill cooking modes?"

### Invalid Queries (Expected to Refuse)
5. "What do you know about the DIALX community?"
6. "What do you think about the dinosaur era? Why did they die?"

## How the RAG System Works

The implementation demonstrates the complete RAG pipeline:

1. **🔍 Retrieval**: Searches for relevant chunks from the microwave manual using FAISS vector similarity search
2. **🔗 Augmentation**: Combines retrieved context with the user question in a structured prompt
3. **🤖 Generation**: Uses GPT-4o to generate an accurate answer based on the provided context

## Implementation Details

All TODO tasks have been completed:
- ✅ Vector store setup with FAISS index caching
- ✅ Document processing with RecursiveCharacterTextSplitter
- ✅ Context retrieval with similarity search
- ✅ Prompt augmentation with RAG template
- ✅ Answer generation with Azure OpenAI (via DIAL)
- ✅ Main configuration with proper embeddings and LLM clients

## Troubleshooting

If you encounter any issues:
1. Ensure the virtual environment is activated
2. Verify dependencies are installed: `pip install -r requirements.txt`
3. Check that the API_KEY is set in `task/_constants.py`
4. Verify VPN connection to EPAM network
5. The FAISS index is automatically created on first run and cached for future use

