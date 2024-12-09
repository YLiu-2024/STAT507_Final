'''
This code is for model downloading
'''
from transformers import AutoTokenizer, AutoModel

def download_and_save_model(dir_path, model_name):
    """
    Download and save a Hugging Face model and tokenizer locally.
    dir_path: The directory where the model and tokenizer will be saved.
    model_name: The name of the model to download from Hugging Face.
    """
    # Load the tokenizer and model from the specified model name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # Save the tokenizer and model to the specified directory
    tokenizer.save_pretrained(dir_path)
    model.save_pretrained(dir_path)

# Example usage
download_and_save_model("./ibm/knowgl-large", "ibm/knowgl-large")
download_and_save_model("./rebel-large", "Babelscape/rebel-large")

