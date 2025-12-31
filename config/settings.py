from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
import os

load_dotenv()

@dataclass
class Settings:
    """
    This class represents the centralized configuration container for the application.
    It loads all required configuration values from environment variables and exposes
    them as strongly typed attributes for use across the system.
    """

    BASE_DIR = Path(__file__).resolve().parent.parent

    _FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS"))
    GPT_MODEL_NAME: str = str(os.getenv("GPT_MODEL_NAME"))
    GROQ_API_KEY: str = str(os.getenv("GROQ_API_KEY"))
    TEMPRATURE: int = int(os.getenv("TEMPRATURE"))
    TAVILY_API_KEY: str = str(os.getenv("TAVILY_API_KEY"))
    TOP_K_WEB_RESULTS: int = int(os.getenv("TOP_K_WEB_RESULTS"))

settings = Settings()
