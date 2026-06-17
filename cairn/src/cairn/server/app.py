from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles

from cairn import __version__
from cairn.server import db
from cairn.server.routers import export, hints, intents, projects, settings

STATIC_DIR = Path(__file__).parent / "static"
ADMIN_TOKEN = os.environ.get("CAIRN_ADMIN_TOKEN", "")

# Endpoints that require admin token (prevents workers from reading other projects)
PROTECTED_PATHS = ("/projects",)
# These sub-paths are exempt (workers need them for fact/intent reporting)
EXEMPT_SUFFIXES = (
    "/heartbeat", "/claim", "/release", "/conclude", "/complete",
    "/intents", "/facts", "/hints",
)


class AdminTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if ADMIN_TOKEN and request.url.path.startswith("/projects"):
            path = request.url.path
            # Allow POST to specific action endpoints (worker needs these for fact/intent reporting)
            if request.method == "POST" and any(path.endswith(s) for s in EXEMPT_SUFFIXES):
                return await call_next(request)
            # All other /projects requests need admin token
            token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if token != ADMIN_TOKEN:
                return Response(status_code=403, content="Forbidden")
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.configure(db.DEFAULT_DB)
    yield


app = FastAPI(
    title="Cairn",
    description="Fact-graph based collaborative exploration protocol",
    version=__version__,
    lifespan=lifespan,
)

if ADMIN_TOKEN:
    app.add_middleware(AdminTokenMiddleware)

app.include_router(settings.router)
app.include_router(projects.router)
app.include_router(hints.router)
app.include_router(intents.router)
app.include_router(export.router)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
