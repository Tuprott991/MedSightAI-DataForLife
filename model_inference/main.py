from fastapi import FastAPI
from analysis import router as analysis_router

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"message": "Hello, FastAPI is working!"}

# Đăng ký router inference
app.include_router(analysis_router, prefix="/api/v1")