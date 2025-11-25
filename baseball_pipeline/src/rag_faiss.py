# src/rag_faiss.py

from __future__ import annotations

import unicodedata
import re
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# 프로젝트 루트 기준 FAISS 디렉토리
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "faiss_index"


def clean_text(text: str | None) -> str:
    """제목 비교를 위한 강력한 정제 함수 (네가 쓰던 버전 그대로)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", text)


def load_faiss_db(
    db_path: str | Path | None = None,
    embedding_model: str = "text-embedding-3-large",
) -> FAISS:
    """
    저장된 FAISS 벡터스토어를 로드.
    - db_path: faiss_index 디렉토리 (index.faiss, index.pkl 이 있는 곳)
    - embedding_model: OpenAI 임베딩 모델 이름
    """
    db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH

    if not db_path.exists():
        raise FileNotFoundError(f"FAISS 디렉토리를 찾을 수 없습니다: {db_path}")

    # OpenAIEmbeddings 는 query 임베딩에만 사용됨. (index는 이미 저장되어 있음)
    embeddings = OpenAIEmbeddings(model=embedding_model)

    vector_db = FAISS.load_local(
        str(db_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_db


def load_full_match_log(
    target_match: str,
    db_path: str | Path | None = None,
    embedding_model: str = "text-embedding-3-large",
) -> str:
    """
    FAISS docstore 전체를 뒤져서 metadata['match_title'] 에
    target_match 가 포함된 모든 조각을 합쳐 하나의 텍스트로 반환.
    - target_match: 예) "2025 한국시리즈 1차전"
    """
    vector_db = load_faiss_db(db_path=db_path, embedding_model=embedding_model)

    target_clean = clean_text(target_match)

    all_docs = vector_db.docstore._dict
    matched_chunks = []

    for _, doc in all_docs.items():
        db_title = doc.metadata.get("match_title", "")
        if target_clean in clean_text(db_title):
            matched_chunks.append(doc.page_content)

    if not matched_chunks:
        raise ValueError(f"해당 경기의 데이터를 찾을 수 없습니다: {target_match}")

    full_text = "\n\n".join(matched_chunks)
    return full_text
