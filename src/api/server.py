from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import time
from src.inferencer import TicketInferencer

app = FastAPI(title="Ticket Classifier POC")
STATIC_DIR = Path("src/api/static")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# instantiate inferencer once
inferencer = TicketInferencer()

# Permitir origens externas para teste (em produção restrinja este valor)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],               # em produção, substituir por lista de domínios confiáveis
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))

@app.post("/api/classify")
async def api_classify(request: Request):
    payload = await request.json()
    # expected payload keys: text, customer_segment (optional), channel (optional)
    ticket = {
        "text": payload.get("text", ""),
        "customer_segment": payload.get("customer_segment", ""),
        "channel": payload.get("channel", "")
    }
    start = time.perf_counter()
    category, priority = inferencer.classify_ticket(ticket)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return JSONResponse({
        "category": str(category),
        "priority": str(priority),
        "latency_ms": round(elapsed_ms, 2)
    })