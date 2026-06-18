# Personal Knowledge Chatbot

A RAG chatbot that lets you chat with your own PDF documents. Built as a learning project to understand the full RAG pipeline, from document ingestion to vector retrieval to LLM response generation.

## What it does

Upload one or more PDFs and ask questions about their content. The chatbot retrieves the most relevant chunks and uses an LLM to generate answers grounded in that context. Conversation memory persists across turns within a session and resets on page refresh.

## How it works

```
PDF -> chunks -> embeddings -> ChromaDB
                                  |
Question -> embedding -> similarity search -> top-k chunks -> LLM -> answer
```

1. **Ingestion**: PDFs are loaded and split into overlapping chunks using `RecursiveCharacterTextSplitter`
2. **Embedding**: Each chunk is embedded using `all-MiniLM-L6-v2` (HuggingFace sentence-transformers)
3. **Storage**: Embeddings go into an in-memory ChromaDB vector store
4. **Retrieval**: At query time, the question is embedded and the top-4 most similar chunks are retrieved
5. **Generation**: Retrieved chunks plus the last 3 conversation turns are passed to Groq's LLaMA 3.1 with a prompt that restricts answers to the provided context
6. **Memory**: Chat history is scoped per session using Gradio's `State` — resets on refresh, no cross-user bleed

## Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (in-memory) |
| LLM | Groq API (LLaMA 3.1 8B Instant) |
| UI | Gradio |
| Deployment | HuggingFace Spaces |

## Things I learned building this

- Chunk size matters more than I expected. Too large and retrieval loses precision; too small and individual chunks lack enough context to be useful.
- Local LLM inference on CPU is not viable for anything interactive. I started with Ollama, got 141-second response times, and switched to Groq.
- ChromaDB's `.persist()` method was removed in newer versions. It auto-persists now, which is fine, just not documented anywhere obvious.
- `ConversationBufferMemory` was removed in LangChain 1.x. Manual chat history scoped through Gradio `State` works fine and is maybe 20 lines.
- HuggingFace Spaces needs secrets configured separately in the UI. `.env` files do not transfer with the push.
- Global state is a real bug in multi-user deployments. Retriever and chat history both need to be session-scoped or users overwrite each other silently.

## Run locally

```bash
git clone https://github.com/ishtdubey/rag-chatbot
cd rag-chatbot
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
```

Run:

```bash
python app.py
```

## Requirements

```
langchain
langchain-community
langchain-chroma
langchain-groq
langchain-huggingface
sentence-transformers
pypdf
gradio
python-dotenv
```

## Live demo

[huggingface.co/spaces/ishtdubey/rag-chatbot](https://huggingface.co/spaces/ishtdubey/rag-chatbot)

## What's next

- LangGraph-based agent with tool use (separate project)
- Better chunking strategies (semantic vs. recursive)
- Hybrid retrieval (BM25 + vector search)
