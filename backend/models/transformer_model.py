import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from models.device import get_device

def train_transformer_model():
    device_str = str(get_device())
    print(f"Using device: {device_str}")
    
    # Reverting to the Bi-Encoder architecture as requested.
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device_str)

    print("Loading Quora Question Pairs dataset...")
    # NOTE: Using a restricted subset (e.g. train[:200]) to quickly verify the pipeline end-to-end.
    # To run a full training on the M4/MPS or CUDA GPU, remove the `split` slice.
    # dataset = load_dataset("glue", "qqp", split="train[:200]")
    dataset = load_dataset("glue", "qqp")["train"]
    train_examples = []
    for row in dataset:
        # QQP labels: 0 = not paraphrase, 1 = paraphrase 
        train_examples.append(InputExample(texts=[row['question1'], row['question2']], label=float(row['label'])))

    # Define DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Switching to standard ContrastiveLoss. 
    # (OnlineContrastiveLoss dynamically mines hard negatives, which involves extremely complex tensor sorting 
    # that currently triggers a known bug in Apple Silicon's MPS GPU driver during gradient calculation).
    train_loss = losses.ContrastiveLoss(model=model, margin=0.5)

    print("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=20,
        warmup_steps=2000,
        show_progress_bar=True
    )
    
    # Save the model
    output_path = os.path.join(os.path.dirname(__file__), 'saved_model')
    print(f"Saving fine-tuned model to {output_path}...")
    model.save(output_path)
    print("Transformer fine-tuning verified and completed.")

if __name__ == "__main__":
    train_transformer_model()
