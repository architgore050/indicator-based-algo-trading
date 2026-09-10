"""FastAPI application entry point."""

import sys
from pathlib import Path

# Ensure v6 directory is importable
v6_dir = Path(__file__).parent.parent
if str(v6_dir) not in sys.path:
    sys.path.insert(0, str(v6_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router

app = FastAPI(
    title="XAUUSD Trading Analytics Dashboard",
    description="API for the XAUUSD algorithmic backtesting analytics dashboard",
    version="1.0.0",
)

# CORS for Vite dev server and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="")


@app.on_event("startup")
def _preload_on_startup():
    """Eagerly load all non-indicator data into RAM so first chart paint is fast.
    Indicator computation remains on-demand (and is fully cached once computed)."""
    try:
        from backend.api.data_loader import preload_all, get_data_bounds
        stats = preload_all()
        bounds = get_data_bounds()
        print(f"[preload] bounds={bounds} stats={stats}")
    except Exception as e:
        print(f"[preload] failed: {e}")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/bounds")
def data_bounds():
    from backend.api.data_loader import get_data_bounds
    return get_data_bounds()


# Mount static files for frontend (production)
frontend_dist = v6_dir / "Analytics dashboard design" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
