from fastapi import FastAPI

from app.api.routes import router
from app.api.right_routes import router as right_routes

app = FastAPI(title="CloudX Access Diagnosis Agent", version="0.1.0")
app.include_router(router)
app.include_router(right_routes)
app.include_router(__import__("app.api.right_routes_v2", fromlist=["router"]).router)
app.include_router(__import__("app.api.right_routes_v3", fromlist=["router"]).router)
app.include_router(__import__("app.api.right_routes_v4", fromlist=["router"]).router)
app.include_router(__import__("app.api.right_routes_v5", fromlist=["router"]).router)
