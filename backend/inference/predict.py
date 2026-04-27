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
            base_path = os.path.dirname(os.path.dirname(__file__)) # Go up to backend/
            v2_path = os.path.join(base_path, 'models', 'saved_model_v2')
            mpnet_path = os.path.join(base_path, 'models', 'saved_model_mpnet')
            minilm_path = os.path.join(base_path, 'models', 'saved_model_minilm')
            
            model_name_or_path = "all-mpnet-base-v2"
            if os.path.isdir(v2_path):
                model_name_or_path = v2_path
            elif model_type == 'mpnet' and os.path.isdir(mpnet_path):
                model_name_or_path = mpnet_path
            elif model_type == 'minilm' and os.path.isdir(minilm_path):
                model_name_or_path = minilm_path
            elif os.path.isdir(os.path.join(base_path, 'models', 'saved_model')):
                model_name_or_path = os.path.join(base_path, 'models', 'saved_model')
                
            _bi_model = SentenceTransformer(model_name_or_path, device=device_str)
            print(f"Bi-Encoder successfully loaded from: {model_name_or_path}")
        except Exception as e:
            print(f"Failed to load Bi-Encoder: {e}")

    if _cross_model is None:
        try:
            # STS-B model: trained on general text (news, captions, forums)
            # unlike quora-distilroberta-base which only understands questions
            print(f"Loading Cross-Encoder (STS-B) to {device_str}...")
            _cross_model = CrossEncoder('cross-encoder/stsb-distilroberta-base', device=device_str)
            print("Cross-Encoder loaded.")
        except Exception as e:
            print(f"Failed to load Cross-Encoder: {e}")

def predict_similarity(text1: str, text2: str) -> tuple[float, bool]:
    """
    Improved hybrid approach:
    1. Bi-Encoder provides fast semantic similarity.
    2. Cross-Encoder (STS-B) provides refined similarity on both sentences together.
    3. Weighted combination makes final decision.
    """
    if _bi_model is None or _cross_model is None:
        initialize_models()
        
    # Step 1: Bi-Encoder cosine similarity
    emb1 = _bi_model.encode(text1, convert_to_tensor=True)
    emb2 = _bi_model.encode(text2, convert_to_tensor=True)
    bi_score = util.cos_sim(emb1, emb2).item()
    
    # Step 2: Cross-Encoder for refinement (only when there's some similarity)
    if bi_score > 0.55:
        cross_raw = _cross_model.predict([text1, text2])
        # STS-B distilroberta outputs scores already in 0-1 range
        cross_score = max(0.0, min(1.0, float(cross_raw)))
        
        print(f"DEBUG: Bi-Score: {bi_score:.4f}, Cross-Score: {cross_score:.4f}")
        
        # Weighted ensemble: Bi-Encoder 35%, Cross-Encoder 65%
        # Optimized via grid search on MRPC validation set (80.88% acc, 0.87 F1)
        final_score = 0.35 * bi_score + 0.65 * cross_score
        
        # Decision threshold (optimized)
        is_paraphrase = final_score > 0.70
    else:
        # Very low bi-score → definitely not a paraphrase, skip cross-encoding
        final_score = bi_score
        is_paraphrase = False
    
    return final_score, is_paraphrase
