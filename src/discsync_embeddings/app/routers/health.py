# external
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
async def health() -> dict:
    """Health check endpoint to verify the service is running."""

    return {"status": "ok"}
