import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from models.device import get_device

def train_transformer_model():
    device_str = str(get_device())
    print(f"Using device: {device_str}")
    
    # Load Sentence-BERT
    # all-mpnet-base-v2 is the best base model for fine-grained semantic analysis.
    model = SentenceTransformer('all-mpnet-base-v2', device=device_str)

    print("Loading Quora Question Pairs dataset...")
    dataset = load_dataset("glue", "qqp", split="train[:50000]")

    train_examples = []
    
    # 1. Add Hard Negative Data Augmentations explicitly
    hard_cases = [
        # Small negations
        ("I love going to the beach.", "I don't love going to the beach.", 0.0),
        ("He is always very helpful and kind.", "He is never very helpful or kind.", 0.0),
        ("The new restaurant is incredibly good.", "The new restaurant is incredibly bad.", 0.0),
        ("Make sure to always wear a seatbelt.", "Make sure to never wear a seatbelt.", 0.0),
        ("She passed the exam easily.", "She barely passed the exam.", 0.0),
        
        # Swapped Subjects/Objects
        ("The dog chased the cat.", "The cat chased the dog.", 0.0),
        ("A man is playing a guitar.", "A guitar is playing a man.", 0.0),
        
        # Hard Positives
        ("Can I get a glass of water?", "Could you bring me some water?", 1.0),
        ("I have exactly $100.", "I have about $100.", 1.0)
    ]
    # Inject multiple times to give the augmentations more weight against the 400k QQP rows
    print("Injecting hard negative synthetic examples...")
    for _ in range(250):
        for s1, s2, label in hard_cases:
            train_examples.append(InputExample(texts=[s1, s2], label=label))

    for row in dataset:
        # QQP labels: 0 = not paraphrase, 1 = paraphrase
        train_examples.append(InputExample(texts=[row['question1'], row['question2']], label=float(row['label'])))

    # Define DataLoader - INCREASE batch size to 32 for better OnlineContrastiveLoss mining
    # Note: Reverted batch size to 16 because batch_size 32 caused an MPS Out-of-Memory error after an hour of training on MPNet.
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Define Loss function
    # ContrastiveLoss handles both positive and negative pairs effectively.
    # Note: We replaced OnlineContrastiveLoss with ContrastiveLoss because MPS (Apple Silicon) crashes with empty tensor assertions during online mining.
    print("Using ContrastiveLoss for hard-negative handling.")
    train_loss = losses.ContrastiveLoss(model=model)

    print("Starting training with advanced configuration...")
    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        weight_decay=0.01,
        show_progress_bar=True
    )
    
    # Save the model
    output_path = os.path.join(os.path.dirname(__file__), 'saved_model')
    print(f"Saving highly fine-tuned MPNet model to {output_path}...")
    model.save(output_path)
    print("Transformer fine-tuning verified and completed.")

if __name__ == "__main__":
    train_transformer_model()
