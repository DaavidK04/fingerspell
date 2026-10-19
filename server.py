from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def load_index():
    return FileResponse("static/index.html")
