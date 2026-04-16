import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from models.device import get_device

def train_transformer_model():
    device_str = str(get_device())
    print(f"Using device: {device_str}")
    
    # Load Sentence-BERT
    # all-MiniLM-L6-v2 is an excellent baseline for semantic similarity.
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device_str)

    print("Loading Quora Question Pairs dataset...")
    # NOTE: Using a restricted subset (e.g. train[:200]) to quickly verify the pipeline end-to-end.
    # To run a full training on the M4/MPS or CUDA GPU, remove the `split` slice.
    dataset = load_dataset("glue", "qqp", split="train")

    train_examples = []
    for row in dataset:
        # QQP labels: 0 = not paraphrase, 1 = paraphrase
        train_examples.append(InputExample(texts=[row['question1'], row['question2']], label=float(row['label'])))

    # Define DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Define Loss function
    # CosineSimilarityLoss works best when labels are floats [0, 1]
    train_loss = losses.CosineSimilarityLoss(model)

    print("Starting training...")
    # Fine-tune the model
    # Note: 'optimizer_params' defaults to AdamW under the hood in SentenceTransformers.
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=10,
        show_progress_bar=True
    )
    
    # Save the model
    output_path = os.path.join(os.path.dirname(__file__), 'saved_model')
    print(f"Saving fine-tuned model to {output_path}...")
    model.save(output_path)
    print("Transformer fine-tuning verified and completed.")

if __name__ == "__main__":
    train_transformer_model()
