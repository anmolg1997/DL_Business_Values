# Databricks notebook source
# MAGIC %pip install --upgrade transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import psutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")


# Define the model names and their corresponding save paths
model_names = {
    "llama31_8b": "meta-llama/Meta-Llama-3.1-8B"
    # "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    # "bert": "nlptown/bert-base-multilingual-uncased-sentiment",
    # "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    # "albert": "textattack/albert-base-v2-SST-2"
}

# Define the directory where models will be saved
save_directory = "/dbfs/mnt/igenie-blob01/Anmol_AI_dir/model_dir"

# Check if the directory exists, if not create it
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print(f"Created directory: {save_directory}")

auth_token = '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'

# Download and save each model and tokenizer
for model_name, model_identifier in model_names.items():

    print(f"Processing model: {model_name}")
    print_memory_usage()
    
    # Download model
    model = AutoModelForSequenceClassification.from_pretrained(model_identifier, token=auth_token)
    model_save_path = f"{save_directory}/{model_name}-model"
    model.save_pretrained(model_save_path)

    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    tokenizer_save_path = f"{save_directory}/{model_name}-tokenizer"
    tokenizer.save_pretrained(tokenizer_save_path)

    print(f"Model and tokenizer for {model_name} saved to {model_save_path} and {tokenizer_save_path} respectively.")


# COMMAND ----------

# MAGIC %pip install --upgrade pip --quiet
# MAGIC %pip install pyspark==3.1.2 prometheus-client==0.15.0 typing-extensions --quiet
# MAGIC %pip install --upgrade accelerate transformers --quiet
# MAGIC
# MAGIC %pip install jinja2==3.1.4 --quiet
# MAGIC %pip install trl peft bitsandbytes --quiet
# MAGIC %pip install torch torchvision torchaudio --quiet
# MAGIC %pip install triton --quiet  # Removed the version specification
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Import necessary libraries
import os
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.tracking.artifact_utils import get_artifact_uri

from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)
# Define constants
MODEL_NAME = 'meta-llama/Meta-Llama-3.1-8B'
LOCAL_MODEL_DIR = '/dbfs/tmp/meta_llama_model_31_8b'
REGISTERED_MODEL_NAME = 'meta-llama-31-8B'
auth_token = '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'

def load_transformer_model(MODEL_NAME:str):
    """Load the transformer model and tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="float16",
        # quantization_config=bnb_config, 
        token=auth_token
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=auth_token)
    return model, tokenizer

def load_and_save_model(model_name: str, local_model_dir: str):
    """Load the model and tokenizer, then save them locally."""
    model = AutoModelForCausalLM.from_pretrained(model_name,  token=auth_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name,  token=auth_token)
    model.save_pretrained(local_model_dir)
    tokenizer.save_pretrained(local_model_dir)
    return model, tokenizer


def log_and_register_model(local_model_dir: str, registered_model_name: str,  model, tokenizer):
    """Log the model and tokenizer with MLflow and register them."""
    mlflow.start_run()
    # Log the model
    mlflow.pytorch.log_model(pytorch_model=model, artifact_path="meta_llama_model_31_8b_artifact")
    # Log the tokenizer
    mlflow.log_artifacts(local_model_dir, artifact_path="tokenizer")
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    client = MlflowClient()
    # model_uri = f"runs:/{run_id}/meta_llama_model"
    source = get_artifact_uri(run_id=run_id, artifact_path="meta_llama_model_31_8b_artifact")

    try:
        client.create_registered_model(registered_model_name)
    except:
        pass
    client.create_model_version(name=registered_model_name, source=source, run_id=run_id)



def main():
    """Main function to execute the workflow."""
    model, tokenizer = load_and_save_model(MODEL_NAME, LOCAL_MODEL_DIR)
    log_and_register_model(LOCAL_MODEL_DIR, REGISTERED_MODEL_NAME, model, tokenizer)
    print("Model and tokenizer loaded, saved, and registered successfully.")


main()


# COMMAND ----------

mlflow.end_run()

# COMMAND ----------


