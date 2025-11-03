# external
from fastapi import APIRouter

# project
from . import health as health_module


def build_api_router() -> APIRouter:
    """Build API Router for discsync-embeddings."""

    api = APIRouter()
    api.include_router(health_module.router)
    return api
