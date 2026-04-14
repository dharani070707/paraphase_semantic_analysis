import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from models.device import get_device

def evaluate():
    device_str = str(get_device())
    
    finetuned_model_path = os.path.join(os.path.dirname(__file__), 'saved_model')
    print(f"Loading model from '{finetuned_model_path}'...")
    model = SentenceTransformer(finetuned_model_path, device=device_str)

    print("Loading QQP validation dataset (40,000+ pairs)...")
    val_dataset = load_dataset("glue", "qqp", split="validation")
    
    texts_1 = [row['question1'] for row in val_dataset]
    texts_2 = [row['question2'] for row in val_dataset]
    labels = [int(row['label']) for row in val_dataset]

    print("Encoding texts into embeddings (this will take a few seconds)...")
    # Encode with a reasonably large batch size to be fast
    embeddings1 = model.encode(texts_1, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    embeddings2 = model.encode(texts_2, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    
    print("Computing pairwise cosine similarities...")
    # Computes pairwise similarity (not NxN to save memory)
    scores = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()

    # Find the optimal threshold that yields the best F1 score
    best_f1 = 0
    best_acc = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.3, 0.95, 0.05):
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_thresh = thresh

    print("\n" + "="*40)
    print("        EVALUATION RESULTS        ")
    print("="*40)
    print(f"Evaluated on: {len(labels)} unseen validation pairs")
    print(f"Optimal Cosine Threshold: {best_thresh:.2f}")
    print(f"Accuracy: {best_acc:.4f}  ({(best_acc*100):.2f}%)")
    print(f"F1 Score: {best_f1:.4f}  ({(best_f1*100):.2f}%)")
    print("="*40)

if __name__ == '__main__':
    evaluate()
