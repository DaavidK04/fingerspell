import logging

from fastapi import FastAPI, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.websockets import WebSocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def load_index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
        except WebSocketDisconnect:
            logger.info("connection closed.")
            return
        else:
            logger.info("frame received: %d bytes", len(data))
