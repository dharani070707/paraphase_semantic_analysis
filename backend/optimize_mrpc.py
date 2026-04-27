import os
from datasets import load_dataset
from inference import predict as inference
from sentence_transformers import util
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def optimize_thresholds():
    print("\n" + "="*50)
    print("  Optimizing Weighted Ensemble for MRPC  ")
    print("="*50)

    inference.initialize_models()

    print("Loading MRPC validation dataset...")
    dataset = load_dataset("glue", "mrpc", split="validation")
    texts1, texts2, labels = dataset['sentence1'], dataset['sentence2'], dataset['label']

    print("Extracting raw scores from Bi-Encoder and Cross-Encoder...")
    bi_scores = []
    cross_scores = []

    for t1, t2 in tqdm(zip(texts1, texts2), total=len(labels)):
        # 1. Bi-Encoder Score
        emb1 = inference._bi_model.encode(t1, convert_to_tensor=True)
        emb2 = inference._bi_model.encode(t2, convert_to_tensor=True)
        bi_scores.append(util.cos_sim(emb1, emb2).item())

        # 2. Cross-Encoder Score (STS-B distilroberta outputs 0-1 directly)
        raw = inference._cross_model.predict([t1, t2])
        cross_scores.append(max(0.0, min(1.0, float(raw))))

    bi_scores = np.array(bi_scores)
    cross_scores = np.array(cross_scores)
    labels = np.array(labels)

    # Grid Search over weight and threshold
    best_acc = 0
    best_f1 = 0
    best_bi_weight = 0
    best_threshold = 0

    print("\nSearching for optimal weight + threshold...")
    for bi_weight in np.arange(0.25, 0.55, 0.05):
        cross_weight = 1.0 - bi_weight
        for threshold in np.arange(0.45, 0.80, 0.025):
            preds = []
            for b, c in zip(bi_scores, cross_scores):
                if b > 0.55:
                    combined = bi_weight * b + cross_weight * c
                    preds.append(1 if combined > threshold else 0)
                else:
                    preds.append(0)
            
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            # Optimize for F1 (better balance of precision/recall)
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_bi_weight = bi_weight
                best_threshold = threshold

    print("\n" + "-"*40)
    print(f"OPTIMAL BI-WEIGHT      : {best_bi_weight:.2f}")
    print(f"OPTIMAL CROSS-WEIGHT   : {1.0 - best_bi_weight:.2f}")
    print(f"OPTIMAL THRESHOLD      : {best_threshold:.3f}")
    print(f"MAX ACCURACY ACHIEVED  : {best_acc:.4f} ({(best_acc*100):.2f}%)")
    print(f"MAX F1 ACHIEVED        : {best_f1:.4f}")
    print("-"*40)
    print(f"\nCopy these into predict.py:")
    print(f"  final_score = {best_bi_weight:.2f} * bi_score + {1.0 - best_bi_weight:.2f} * cross_score")
    print(f"  is_paraphrase = final_score > {best_threshold:.3f}")
    print()

if __name__ == "__main__":
    optimize_thresholds()
