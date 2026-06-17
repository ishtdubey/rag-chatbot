import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
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
rag_chain = None
retriever = None


# ====================== Initialize / Update Vector Store ======================
def initialize_vector_store(new_splits=None):
    global vectorstore, rag_chain, retriever

    if vectorstore is None:
        if os.path.exists(VECTOR_STORE_PATH) and len(os.listdir(VECTOR_STORE_PATH)) > 0:
            print("Loading existing vector store...")
            vectorstore = Chroma(
                persist_directory=VECTOR_STORE_PATH,
                embedding_function=embeddings
            )
        else:
            print("Creating new vector store...")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=VECTOR_STORE_PATH
            )

    if new_splits:
        print(f"Adding {len(new_splits)} new chunks...")
        vectorstore.add_documents(new_splits)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    template = """Answer the question based only on the following context.
If you don't know the answer, say "I don't have enough information."

Context: {context}
Question: {question}
Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


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
    return f"Done. {len(all_splits)} chunks embedded. You can now ask questions."


# ====================== Chat ======================
def chat(message, history):
    global rag_chain, retriever

    if rag_chain is None:
        return "Please upload at least one PDF first."

    try:
        response = rag_chain.invoke(message)

        docs = retriever.invoke(message)
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


# ====================== Main ======================
if __name__ == "__main__":
    if os.path.exists(VECTOR_STORE_PATH) and len(os.listdir(VECTOR_STORE_PATH)) > 0:
        print("Loading existing vector store on startup...")
        initialize_vector_store()
    else:
        print("No existing vector store. Upload PDFs to get started.")

    with gr.Blocks(title="Personal Knowledge Chatbot") as demo:
        gr.Markdown("# Personal Knowledge Chatbot")
        gr.Markdown("Upload your PDFs first, then ask questions about their content.")

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
            description="Ask any question about your uploaded documents.",
        )

    demo.launch(share=False)