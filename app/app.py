# Refer to https://github.com/shaaagri/iat481-nlp-proj/blob/main/Llama2_RAG_bot.ipynb notebook where
# we went over our code in detail

import yaml
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.globals import set_debug


prompt_template="""[INST]
<<SYS>>
You are helpful, respectful, caring and honest assistant for question-answering tasks. You do not have expressions or emotions. You are objective and provide everything that is helpful to know given the question, but you are not chatty, be concise and do not use more than three sentences. Use the following pieces of retrieved context to answer the question to the best of your ability. If you don't know the answer, just say that you don't know.
<</SYS>>

USER: {question}

CONTEXT: {context}

ASSISTANT: 
[/INST]
"""


def load_config():
    global config

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)


def load_model(model_name_or_path, model_basename):
    global model_path
    # Downloads Llama-2 upon accessing it for the first time. Otherwise, loads it from cache
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

  
def init_llama():
    global prompt, llm

    prompt = PromptTemplate.from_template(prompt_template)

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        # Make sure the model path is correct for your system!
        model_path=model_path,
        
        temperature=0.6,
        n_gpu_layers=-1,  # -1 stands for offloading all layers of the model to GPU => better performance (we've got enough VRAM)
        n_ctx=4096,  # IMPORTANT for RAG. the default for quantized GGUF models is only 512
        max_tokens=1024,
        repeat_penalty=1.02,
        top_p=0.8, # nucleus sampling
        top_k=150,  # sample from k top tokens 
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )  

    set_debug(config['debug']) 


def main():
    load_config()
    load_model(config['model_name_or_path'], config['model_basename'])
    init_llama()

    embedding_function = SentenceTransformerEmbeddings(model_name=config['embedding_model'])

    
    question='Describe the main campus of the Simon Fraser University'

    llm.invoke(prompt.format(question=question, context=''))

    print("yo")


if __name__ == "__main__":
    main()

