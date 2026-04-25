import os
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from models.device import get_device

_bi_model = None
_cross_model = None

def initialize_models(model_type='mpnet'):
    """
    Loads both the Bi-Encoder (for speed) and Cross-Encoder (for precision).
    """
    global _bi_model, _cross_model
    device_str = str(get_device())
    
    if _bi_model is None:
        try:
            print(f"Loading Bi-Encoder ({model_type}) to {device_str}...")
            base_path = os.path.dirname(__file__)
            mpnet_path = os.path.join(base_path, 'saved_model_mpnet')
            minilm_path = os.path.join(base_path, 'saved_model_minilm')
            
            model_name_or_path = "all-mpnet-base-v2"
            if model_type == 'mpnet' and os.path.isdir(mpnet_path):
                model_name_or_path = mpnet_path
            elif model_type == 'minilm' and os.path.isdir(minilm_path):
                model_name_or_path = minilm_path
            elif os.path.isdir(os.path.join(base_path, 'saved_model')):
                model_name_or_path = os.path.join(base_path, 'saved_model')
                
            _bi_model = SentenceTransformer(model_name_or_path, device=device_str)
            print("Bi-Encoder loaded.")
        except Exception as e:
            print(f"Failed to load Bi-Encoder: {e}")

    if _cross_model is None:
        try:
            print(f"Loading Cross-Encoder (Quora) to {device_str}...")
            _cross_model = CrossEncoder('cross-encoder/quora-distilroberta-base', device=device_str)
            print("Cross-Encoder loaded.")
        except Exception as e:
            print(f"Failed to load Cross-Encoder: {e}")

def predict_similarity(text1: str, text2: str) -> tuple[float, bool]:
    """
    Hybrid approach:
    1. Quick check with Bi-Encoder.
    2. If similarity is high (>0.80), verify with Cross-Encoder to catch traps.
    """
    if _bi_model is None or _cross_model is None:
        initialize_models()
        
    emb1 = _bi_model.encode(text1, convert_to_tensor=True)
    emb2 = _bi_model.encode(text2, convert_to_tensor=True)
    bi_score = util.cos_sim(emb1, emb2).item()
    
    # Optimized thresholds for better general performance
    if bi_score > 0.65:
        cross_score = _cross_model.predict([text1, text2])
        print(f"DEBUG: Bi-Score: {bi_score:.4f}, Cross-Score: {cross_score:.4f}")
        
        # Heuristic for cross-encoders
        # Tuned to 0.30 as a safe middle ground for production
        is_paraphrase = (bi_score > 0.65) and (cross_score > 0.30)
        final_score = bi_score if is_paraphrase else min(bi_score, 0.4)
    else:
        is_paraphrase = bi_score >= 0.65
        final_score = bi_score
    
    return final_score, is_paraphrase
