import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
import gradio as gr
import shutil
from config import *

# ====================== CONFIG ======================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

llm = ChatGroq(
    model=LLM_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

vectorstore = None
qa_chain = None


# ====================== Reset Vector Store on Startup ======================
def reset_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    print("Vector store reset.")


vectorstore = None
retriever = None
chat_history = []

def initialize_vector_store(new_splits):
    global vectorstore, retriever

    vectorstore = Chroma.from_documents(
        documents=new_splits,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


def chat(message, history):
    global retriever, chat_history

    if retriever is None:
        return "Please upload at least one PDF first."

    try:
        # Build context from retrieved docs
        docs = retriever.invoke(message)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Build chat history string
        history_text = ""
        for human, assistant in chat_history[-3:]:  # last 3 turns only
            history_text += f"Human: {human}\nAssistant: {assistant}\n"

        template = f"""Answer the question based only on the following context.
If you don't know the answer, say "I don't have enough information."

Context: {context}

Previous conversation:
{history_text}
Human: {message}
Assistant:"""

        response = llm.invoke(template).content

        # Update memory
        chat_history.append((message, response))

        # Sources
        unique_sources = set()
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            filename = source.split('\\')[-1].split('/')[-1]
            unique_sources.add(filename)

        sources_text = "\n".join(f"• {src}" for src in list(unique_sources)[:5])
        return response + f"\n\n**Sources:**\n{sources_text}"

    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}"


# ====================== Upload & Embed ======================
def upload_pdfs(files):
    if not files:
        return "No files uploaded."

    all_splits = []
    for file in files:
        try:
            loader = PyPDFLoader(file)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
            print(f"Loaded {len(splits)} chunks from {file}")
        except Exception as e:
            return f"Error processing file: {str(e)}"

    if not all_splits:
        return "No readable content found in uploaded PDFs."

    initialize_vector_store(all_splits)
    return f"✅ Done. {len(all_splits)} chunks embedded. You can now ask questions."

# ====================== Main ======================
if __name__ == "__main__":
    reset_vector_store()  # Always reset on startup

    with gr.Blocks(title="Personal Knowledge Chatbot") as demo:
        gr.Markdown("# Personal Knowledge Chatbot")
        gr.Markdown("Upload your PDFs first, then ask questions. Memory is maintained across the conversation.")

        with gr.Row():
            file_input = gr.File(
                file_count="multiple",
                label="Upload PDFs",
                type="filepath"
            )
            upload_status = gr.Textbox(
                label="Upload Status",
                interactive=False,
                placeholder="Status will appear here after upload..."
            )

        upload_btn = gr.Button("Upload & Embed PDFs", variant="primary")
        upload_btn.click(
            fn=upload_pdfs,
            inputs=file_input,
            outputs=upload_status
        )

        gr.Markdown("---")

        gr.ChatInterface(
            fn=chat,
            title="Chat with your Documents",
            description="Ask follow-up questions freely, the chatbot remembers the conversation.",
        )

    demo.launch(share=False)