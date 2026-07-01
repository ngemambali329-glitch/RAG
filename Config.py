"""
Central configuration for the Power Systems RAG assistant (Groq version).
Loads settings from .env (falls back to sane defaults).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"              # put your PDFs / txt / md source docs here
DB_DIR = BASE_DIR / "chroma_db"           # persistent vector store lives here
COLLECTION_NAME = "power_systems_engineering"

# Groq: free-tier hosted inference, no local install/storage needed.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
TOP_K = int(os.getenv("TOP_K", 5))

# System prompt that shapes the assistant's teaching persona.
SYSTEM_PROMPT = """You are an expert power systems engineering tutor and research assistant.
You answer using the retrieved course/reference material provided as CONTEXT below, combined
with sound engineering judgment. Your audience is engineering students and practicing engineers
studying topics such as per-unit systems, load flow analysis, symmetrical components, fault
analysis, protection coordination, power system stability, transformers, transmission line
modeling, and power electronics for grids.

Rules:
1. Ground your answer in the provided CONTEXT whenever it is relevant. If the context doesn't
   cover the question, say so explicitly, then answer from general power systems engineering
   knowledge, clearly flagged as "(not from your uploaded materials)".
2. Show working for any calculation (state assumptions, formulas, units, per-unit base values).
3. Use correct engineering notation (e.g., per-unit, p.u., MVA, kV, pu impedance, pf) and SI units.
4. When helpful, explain concepts the way a patient tutor would: build intuition first, then
   formalize with equations.
5. Cite which source chunk(s) your answer draws from, e.g. [Source: filename.pdf, chunk 3].
6. If asked to quiz the student, generate original practice questions rather than copying text
   verbatim from the context.
"""
