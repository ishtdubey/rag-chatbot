import os
from dotenv import load_dotenv

load_dotenv()

# config
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
DATA_FOLDER = os.getenv("DATA_FOLDER", "documents")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)