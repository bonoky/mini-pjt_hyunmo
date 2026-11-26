"""공통 설정: 수업과 같은 Amazon Bedrock 모델을 씁니다."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# 평가 재현성: 같은 URL 은 data/cache 에 저장된 응답을 재사용 (TECHSCOUT_CACHE=0 이면 항상 새로 조회)
CACHE_DIR = ROOT / "data" / "cache"
USE_CACHE = os.getenv("TECHSCOUT_CACHE", "1") == "1"

KST = timezone(timedelta(hours=9))


def today() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


@lru_cache(maxsize=1)
def get_llm():
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=REGION)


def text_of(content) -> str:
    """ChatBedrockConverse 는 content 를 블록 리스트로 주기도 하므로 텍스트만 모읍니다."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)
