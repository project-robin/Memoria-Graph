from __future__ import annotations

from google import genai

from multimodal_search.config import get_api_key


def create_genai_client() -> genai.Client:
    return genai.Client(api_key=get_api_key())
