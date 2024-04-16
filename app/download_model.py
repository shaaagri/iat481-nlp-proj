import yaml
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Execute this line to download the LLM for the first time
model_path = hf_hub_download(repo_id=config['model_name_or_path'], filename=config['model_basename'])
