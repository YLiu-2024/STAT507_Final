
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import json
import torch

def main():
    # 1. Load the dataset
    def load_custom_dataset(train_path, test_path):
        def read_jsonl(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]

        train_data = read_jsonl(train_path)
        test_data = read_jsonl(test_path)

        return train_data, test_data

    # Load the dataset in Rebel format
    train_data, test_data = load_custom_dataset("train_rebel.json", "test_rebel.json")

    # Convert to Hugging Face Dataset format
    def prepare_hf_dataset(data):
        return {
            "id": [item["id"] for item in data],
            "input_text": [item["context"] for item in data],
            "target_text": [item["triplets"] for item in data]
        }

    hf_train_dataset = Dataset.from_dict(prepare_hf_dataset(train_data))
    hf_test_dataset = Dataset.from_dict(prepare_hf_dataset(test_data))

    # 2. Load the pre-trained Rebel model and tokenizer
    model_name = "Babelscape/rebel-large"  # Rebel model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 3. Data preprocessing
    def preprocess_function(examples):
        model_inputs = tokenizer(examples["input_text"], max_length=256, truncation=True, padding="max_length")
        labels = tokenizer(examples["target_text"], max_length=256, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = hf_train_dataset.map(preprocess_function, batched=True)
    test_dataset = hf_test_dataset.map(preprocess_function, batched=True)

    # 4. Ensure the model uses the GPU
    # Or ensure the model is on the CPU
    #model.to('cpu')

    # 5. Training settings
    training_args = TrainingArguments(
        output_dir="./rebel-finetuned",  # Path to save the output model
        evaluation_strategy="epoch",  # Evaluate at each epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        gradient_accumulation_steps=8,  # Gradient accumulation to reduce memory usage
        per_device_train_batch_size=1,  # Batch size per device for training
        per_device_eval_batch_size=1,  # Batch size per device for evaluation
        dataloader_num_workers=2,  # Number of DataLoader workers
        num_train_epochs=3,  # Number of training epochs
        weight_decay=0.010,  # Weight decay
        save_total_limit=2,  # Keep only the latest models
        logging_dir="./logs",  # Path to save logs
        logging_steps=100,  # Log at every 100 steps
        fp16=True,  # Use mixed precision
        # use CPU
        #no_cuda=True,
    )

    # 6. Train using the GPU
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # 7. Start training
    trainer.train()

    # Clear GPU memory after training
    torch.cuda.empty_cache()

    # 8. Save the model
    trainer.save_model("./rebel-finetuned")
    tokenizer.save_pretrained("./rebel-finetuned")


if __name__ == "__main__":
    # Set segmented memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
