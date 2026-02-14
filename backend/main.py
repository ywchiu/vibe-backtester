"""
BackTester API 主程式

投資回測系統的後端 API 服務
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import router
from api.ma_routes import router as ma_router

# 載入環境變數
load_dotenv()

# 建立 FastAPI 應用
app = FastAPI(
    title="BackTester API",
    description="投資回測系統 API - 比較不同投資標的的歷史績效",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 設定 - 開發環境允許所有來源
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊路由
app.include_router(router)
app.include_router(ma_router)


@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "Welcome to BackTester API",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
