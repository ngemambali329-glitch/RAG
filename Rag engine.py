"""
rag_engine.py (Groq version)
------------------------------
Core retrieval-augmented generation logic:
  1. Embed the user's question (locally, via sentence-transformers — free, no API)
  2. Retrieve the top-k most relevant chunks from Chroma
  3. Assemble a context-aware prompt
  4. Call Groq's free hosted API to generate a grounded answer (fast, no local storage)
  5. Keep short-term conversation memory for follow-up questions
"""
from dataclasses import dataclass, field
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

import config


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float


@dataclass
class ConversationState:
    history: List[Dict[str, str]] = field(default_factory=list)  # [{role, content}, ...]

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def as_messages(self):
        return list(self.history)


class PowerSystemsRAG:
    def __init__(self):
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your-groq-api-key-here":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env, sign up for a free "
                "key at https://console.groq.com (API Keys -> Create API Key), and paste "
                "it into .env."
            )

        self.client = Groq(api_key=config.GROQ_API_KEY)

        chroma_client = chromadb.PersistentClient(path=str(config.DB_DIR))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        try:
            self.collection = chroma_client.get_collection(
                name=config.COLLECTION_NAME, embedding_function=embed_fn
            )
        except Exception as e:
            raise RuntimeError(
                "No vector collection found. Run `python ingest.py` first to index your "
                "power systems documents."
            ) from e

    # ---------- Retrieval ----------
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        top_k = top_k or config.TOP_K
        results = self.collection.query(query_texts=[query], n_results=top_k)

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, dists):
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", -1),
                    distance=dist,
                )
            )
        return chunks

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant material was found in the knowledge base for this question."
        blocks = []
        for c in chunks:
            blocks.append(
                f"[Source: {c.source}, chunk {c.chunk_index} | relevance score: {1 - c.distance:.3f}]\n{c.text}"
            )
        return "\n\n---\n\n".join(blocks)

    # ---------- Generation ----------
    def ask(self, question: str, state: ConversationState, top_k: int = None) -> Dict:
        chunks = self.retrieve(question, top_k=top_k)
        context = self.format_context(chunks)

        user_turn = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}"
        )

        messages = (
            [{"role": "system", "content": config.SYSTEM_PROMPT}]
            + state.as_messages()
            + [{"role": "user", "content": user_turn}]
        )

        try:
            response = self.client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                max_tokens=1500,
            )
        except Exception as e:
            raise RuntimeError(
                f"Groq API request failed: {e}\n"
                "Check that your GROQ_API_KEY in .env is correct and that you haven't "
                "hit the free-tier rate limit (wait a minute and try again)."
            ) from e

        answer = response.choices[0].message.content

        # Store the plain question (not the big context blob) in history so it doesn't
        # balloon token usage / context length across turns.
        state.add("user", question)
        state.add("assistant", answer)

        return {
            "answer": answer,
            "sources": [
                {"source": c.source, "chunk_index": c.chunk_index, "relevance": round(1 - c.distance, 3)}
                for c in chunks
            ],
        }
