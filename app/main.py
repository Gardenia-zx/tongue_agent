from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from app.agent.graph import compile_agent_graph
from app.api.routes_agent import router as agent_router
from app.api.routes_health import router as health_router
from app.core.config import get_settings
from app.integrations.langgraph_persistence import open_langgraph_persistence


class UTF8ORJSONResponse(ORJSONResponse):
    media_type = "application/json; charset=utf-8"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with open_langgraph_persistence() as persistence:
        app.state.agent_checkpointer = persistence.checkpointer
        app.state.agent_store = persistence.store
        app.state.agent_graph = compile_agent_graph(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
        )
        yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
        default_response_class=UTF8ORJSONResponse,
    )

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(agent_router, prefix="/api/v1")

    return app


app = create_app()
