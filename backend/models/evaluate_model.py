import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from models.device import get_device

def evaluate_model(model_name: str, model_path: str, texts_1, texts_2, labels, device_str):
    print(f"\nEvaluating {model_name}...")
    if not os.path.exists(model_path):
        print(f"Skipping {model_name}: Path '{model_path}' does not exist yet. Run training to create it.")
        return

    print(f"Loading '{model_path}'...")
    model = SentenceTransformer(model_path, device=device_str)

    print("Encoding texts into embeddings...")
    embeddings1 = model.encode(texts_1, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    embeddings2 = model.encode(texts_2, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    
    print("Computing pairwise cosine similarities...")
    scores = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()

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

    print(f"\n--- {model_name} RESULTS ---")
    print(f"Optimal Cosine Threshold: {best_thresh:.2f}")
    print(f"Accuracy: {best_acc:.4f}  ({(best_acc*100):.2f}%)")
    print(f"F1 Score: {best_f1:.4f}  ({(best_f1*100):.2f}%)\n")

def evaluate_all():
    device_str = str(get_device())
    print("Loading QQP validation dataset (40,000+ pairs)...")
    val_dataset = load_dataset("glue", "qqp", split="validation")
    texts_1 = [row['question1'] for row in val_dataset]
    texts_2 = [row['question2'] for row in val_dataset]
    labels = [int(row['label']) for row in val_dataset]

    print("\n" + "="*40)
    print("      SIDE-BY-SIDE EVALUATION      ")
    print("="*40)

    # 1. Evaluate the OLD model (CosineSimilarityLoss)
    old_model_path = os.path.join(os.path.dirname(__file__), 'saved_bi_encoder_model')
    evaluate_model("OLD MODEL (CosineSimilarityLoss)", old_model_path, texts_1, texts_2, labels, device_str)

    # 2. Evaluate the NEW model (OnlineContrastiveLoss)
    new_model_path = os.path.join(os.path.dirname(__file__), 'saved_model')
    evaluate_model("NEW MODEL (OnlineContrastiveLoss)", new_model_path, texts_1, texts_2, labels, device_str)

    print("="*40)

if __name__ == '__main__':
    evaluate_all()
