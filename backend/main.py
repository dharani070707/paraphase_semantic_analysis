from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.inference import initialize_models, predict_similarity

app = FastAPI(
    title="Paraphrase & Semantic Similarity API",
    description="API for checking if two sentences are paraphrases and retrieving their similarity score.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev, or specify ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load machine learning model during startup
@app.on_event("startup")
def startup_event():
    initialize_models()

class AnalyzeRequest(BaseModel):
    text1: str
    text2: str

class AnalyzeResponse(BaseModel):
    similarity: float
    paraphrase: bool

@app.get("/")
def health_check():
    return {"status": "Backend running"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    if not payload.text1.strip() or not payload.text2.strip():
        raise HTTPException(status_code=400, detail="Missing input text")
    
    similarity, is_paraphrase = predict_similarity(payload.text1, payload.text2)
    
    return AnalyzeResponse(
        similarity=round(float(similarity), 4),
        paraphrase=bool(is_paraphrase)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
