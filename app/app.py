# Refer to https://github.com/shaaagri/iat481-nlp-proj/blob/main/Llama2_RAG_bot.ipynb notebook where
# we went over our code in detail

import sys, os
import yaml
from queue import Queue
import threading
from waiting import wait, ANY, ALL
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.globals import set_debug
import gradio as gr


streaming_tokens = None

prompt_template="""[INST]
<<SYS>>
You are helpful, respectful, caring and honest assistant for question-answering tasks.
You do not have expressions or emotions.
You are objective and provide everything that is helpful to know given the question, but you are not chatty.
Be concise and do not use more than five sentences.
Use the following pieces of retrieved context without relying on your own knowledge to answer the question to the best of your ability
If you don't know the answer, just say that you don't know.
Finish your response with a cheerful or optimistic saying for the user and ask them whether they need more help.
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


def init_app(web_mode=False):
    global db
    load_model(config['model_name_or_path'], config['model_basename'])
    init_llama(web_mode)
    db = load_db()

  
def init_llama(web_mode=False):
    global prompt, llm

    prompt = PromptTemplate.from_template(prompt_template)

    # Callbacks support token-wise streaming
    streaming_handler = StreamingStdOutCallbackHandler()

    if web_mode:
        streaming_handler.on_llm_start = on_llm_start
        streaming_handler.on_llm_new_token = on_llm_new_token
        streaming_handler.on_llm_end = on_llm_end

    callback_manager = CallbackManager([streaming_handler])

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


def load_db():
    return Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)


def vectorize_file(file_path):
    # Code based on examples from the LangChain documentation: 
    # https://python.langchain.com/docs/integrations/vectorstores/chroma/

    file_ext = os.path.splitext(file_path)

    if file_ext[1] == '.csv':
        loader = CSVLoader(file_path)
    else:
        loader = TextLoader(file_path)

    print("Ingesting the knowledge data...")
    documents = loader.load()

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    docs = text_splitter.split_documents(documents)
    print(f'{len(docs)} chunks have been created\n')    

    print("Saving chunks to the vector store...")

    try:
        clear_chroma_db(db)  # Empty the database (this line is ignored if it's not been initialized yet)
    except NameError:
        pass

    db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
    print(f'{db._collection.count()} vectors have been added to the store')


def clear_chroma_db(db):
    if db is None:
        return
    
    try:
        db.delete_collection()
    except:
        pass


def main():
    init_app()

    qa_chain = RetrievalQA.from_chain_type(
        llm,

        # When choosing the top_k we try to pick only the most relevant Q&A pairs.
        # Our dataset is small so that should suffice and we won't bloat the prompt
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs={"prompt": prompt}
    )
    
    question = input("\n\nEnter your question for the sleep assistant (leave it blank to cancel): ")
    
    if question:
        qa_chain.invoke(question)


def gradio_predict(question, history):
    qa_chain = RetrievalQA.from_chain_type(
        llm,

        # When choosing the top_k we try to pick only the most relevant Q&A pairs.
        # Our dataset is small so that should suffice and we won't bloat the prompt
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs={"prompt": prompt}
    )

    llm_thread = threading.Thread(target=lambda: qa_chain.invoke(question)).start()

    wait(lambda: has_streaming_started(), timeout_seconds=30, sleep_seconds=0.25)
    
    partial_response = ""

    while streaming_tokens is not None:
        wait(ANY([lambda: is_token_queue_not_empty(), lambda: has_streaming_ended()]), timeout_seconds=30, sleep_seconds=0.1)

        if streaming_tokens is not None:
            for i in range(0, streaming_tokens.qsize()):
                next_token = streaming_tokens.get()
                partial_response += next_token

            yield partial_response

    print("gradio predict finished")


def on_llm_start(serialized, prompts, **kwargs):
    global streaming_tokens
    streaming_tokens = Queue(maxsize = 1024)


def has_streaming_started():
    global streaming_tokens
    return streaming_tokens is not None


def has_streaming_ended():
    global streaming_tokens
    return streaming_tokens is None


def is_token_queue_not_empty():
    global streaming_tokens
    if streaming_tokens is None:
        return False

    return streaming_tokens.qsize() > 0


def on_llm_new_token(token, **kwargs):
    global streaming_tokens
    streaming_tokens.put(token)


def on_llm_end(response, **kwargs):
    global streaming_tokens
    streaming_tokens = None
    print(has_streaming_ended())
    print("llm finished")
   
   
load_config()
embedding_function = SentenceTransformerEmbeddings(model_name=config['embedding_model'])

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'vectorize':
            vectorize_file(sys.argv[2])
            sys.exit(0)

        if sys.argv[1].lower() == 'webui':
            init_app(web_mode=True)
            gr.ChatInterface(gradio_predict).launch()
            sys.exit(0) 

    main()
