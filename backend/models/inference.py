import os
from sentence_transformers import SentenceTransformer, util
from models.device import get_device

_model = None

def initialize_models():
    """
    Loads the machine learning models into memory lazily.
    Checks for locally fine-tuned models first, then falls back to base model.
    """
    global _model
    if _model is None:
        try:
            device_str = str(get_device())
            print(f"Loading transformer model to {device_str}...")
            
            # Use pre-trained all-MiniLM-L6-v2 as a highly capable baseline if fine-tuned weights aren't present yet.
            # You can change the path here once your local training script finishes.
            model_name_or_path = "all-MiniLM-L6-v2"
            
            # Optional: Check if a locally saved model directory exists and use it
            saved_model_path = os.path.join(os.path.dirname(__file__), 'saved_model')
            if os.path.isdir(saved_model_path):
                print(f"Found fine-tuned model at {saved_model_path}, loading it...")
                model_name_or_path = saved_model_path
                
            _model = SentenceTransformer(model_name_or_path, device=device_str)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")

def predict_similarity(text1: str, text2: str) -> tuple[float, bool]:
    """
    Given two strings, returns a tuple of (similarity_score, is_paraphrase).
    """
    if _model is None:
        initialize_models()
        
    embeddings1 = _model.encode(text1, convert_to_tensor=True)
    embeddings2 = _model.encode(text2, convert_to_tensor=True)
    
    cosine_score = util.cos_sim(embeddings1, embeddings2).item()
    
    # We use MiniLM out of the box because it is highly optimized for semantic similarity.
    # Simple thresholding logic: If score > 0.60, it's considered a paraphrase.
    is_paraphrase = cosine_score >= 0.60
    
    return cosine_score, is_paraphrase
