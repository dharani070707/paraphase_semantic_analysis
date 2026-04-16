import os
from sentence_transformers import SentenceTransformer, util
from models.device import get_device

_model = None

def initialize_models(model_type='mpnet'):
    """
    Loads the machine learning models into memory lazily.
    Checks for locally fine-tuned models (mpnet or minilm) first, then falls back to base model.
    """
    global _model
    if _model is None:
        try:
            device_str = str(get_device())
            print(f"Loading transformer model ({model_type}) to {device_str}...")
            
            # Use pre-trained all-mpnet-base-v2 as a highly capable baseline
            model_name_or_path = "all-mpnet-base-v2"
            
            # Check for specific saved model directories
            base_path = os.path.dirname(__file__)
            mpnet_path = os.path.join(base_path, 'saved_model_mpnet')
            minilm_path = os.path.join(base_path, 'saved_model_minilm')
            
            if model_type == 'mpnet' and os.path.isdir(mpnet_path):
                print(f"Found fine-tuned MPNet model at {mpnet_path}, loading it...")
                model_name_or_path = mpnet_path
            elif model_type == 'minilm' and os.path.isdir(minilm_path):
                print(f"Found fine-tuned MiniLM model at {minilm_path}, loading it...")
                model_name_or_path = minilm_path
            elif os.path.isdir(os.path.join(base_path, 'saved_model')): # Backward compatibility
                model_name_or_path = os.path.join(base_path, 'saved_model')
                
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
    # Simple thresholding logic: If score > 0.75, it's considered a paraphrase.
    is_paraphrase = cosine_score > 0.75
    
    return cosine_score, is_paraphrase
