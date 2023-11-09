"""
Steps to Run code:
1. Install requirements "pip install -r requirements.txt"
Note: if you are using CUDA you need to install Torch differently

2. Run "chainlit run example_UI_app.py" in this directory
Note: if you have response issues make sure you have llama_cpp_python==0.2.11
"""
# To run this code paste "chainlit run example_UI_app.py" in this directory

import asyncio
import chromadb
import torch

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import LlamaCPP
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer

from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.prompts  import PromptTemplate
from llama_index.llms import ChatMessage, MessageRole
 

from llama_index.callbacks.base import CallbackManager
import chainlit as cl


QUESTION_COUNT = 0
CHAT_HISTORY = []

QA_TEMPLATE = PromptTemplate(
    """
    You are a "PaperBot", an AI assistant for answering questions about a arXiv paper. Assume all questions you receive are about this paper.
    Please limit your answers to the information provided in the "Context:"

    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context_str}

    Use that context to answer the following question about the paper, including quotes is welcomed.
    Question: {query_str}

    Answer: """)
    
# Create a Standalone Question by combining it with the relevant CHAT HISTORY.
MEM_PROMPT = PromptTemplate("""
    Your objective is to take in the USER QUESTION and fill in the missing context from the CHAT HISTORY.             
    The question is always about an arXiv paper, do not modify acronyms and use FIRST and LAST Name.
    DO NOT CHANGE THE QUESTION WORD: who, what, when, where, why, how.
                            
    Here is an example of your task:
    EXAMPLE START:
        CHAT HISTORY: [ChatMessage(role=<messagerole.user: 'user'>, content='Who was the first President of the US?'), chatmessage(role=<messagerole.assistant: 'assistant'>, content='\n  George Washington.')]
        USER QUESTION: Where did he live?
        STANDALONE QUESTION: Where did George Washington live?
    EXAMPLE END.
    
    CHAT HISTORY: {chat_history}   
                                                    
    USER QUESTION: {question}
                            
    STANDALONE QUESTION: """)


@cl.on_chat_start
async def factory():
    global QA_TEMPLATE, MEM_PROMPT
    # Detect hardware acceleration device
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_layers = 50
    elif torch.backends.mps.is_available():  # Assuming MPS backend exists
        device = 'mps'
        gpu_layers = 1
    else:
        device = 'cpu'
        gpu_layers = 0

    print(f'Using device: {device}')

    embed_model_name = 'BAAI/bge-small-en-v1.5'
    # Create an instance of HuggingFace
    embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        device = device,
        normalize='True'  
        )
    # load from disk
    path = 'RAG_VectorDB'
    db = chromadb.PersistentClient(path=path)

    chroma_collection = db.get_or_create_collection('arxiv_PDF_DB')

    if embed_model_name != chroma_collection.metadata['embedding_used']:
        raise Warning('Not using the same embedding model!')

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    model_url = 'https://huggingface.co/TheBloke/Mistral-11B-OmniMix-GGUF/resolve/main/mistral-11b-omnimix-bf16.Q4_K_M.gguf'

    llm = LlamaCPP(
        # We can pass the URL to a GGUF model to download it 
        model_url=model_url,
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={'n_gpu_layers': gpu_layers},
        verbose=False,
    )

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=llm,
        # callback manager show progress in UI
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context,
        storage_context=storage_context
    )


    # percentile_cutoff: a measure for using the top percentage of relevant sentences.
    query_engine = index.as_query_engine(streaming=True, similarity_top_k = 2, text_qa_template=QA_TEMPLATE,
    node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5, embed_model=embed_model)]
    )
    
    CHAT_HISTORY = []

    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        embed_model=embed_model,
        service_context = service_context,
        condense_question_prompt=MEM_PROMPT,
        chat_history=CHAT_HISTORY,
        verbose=False,
    )

    print('Model Loaded')
    cl.user_session.set('chat_engine', chat_engine)


@cl.on_message
async def main(message: cl.Message):
    global QUESTION_COUNT
    question = message.content
    chat_engine = cl.user_session.get('chat_engine') 
    if QUESTION_COUNT == 0: # Since there is no chat history at the beginning, skip rephrasing with memory
        print('First question')
        response = await cl.make_async(chat_engine._query_engine.query)(question)

        response_message = cl.Message(content='')

        for token in response.response_gen:
            await response_message.stream_token(token=token.replace('  ', ''))
        
        msg_content = "\n\nSources:\n"
        title = response.source_nodes[0].metadata["source"]
        link = response.source_nodes[0].metadata["link"]
        msg_content += f'\u00A0 \- [{title}]({link}) '
        page0 = response.source_nodes[0].metadata["page"]
        page1 = response.source_nodes[1].metadata["page"]
        if page0 == page1: msg_content += f'page: {page0}\n'
        else: msg_content += f'pages: {page0} & {page1}\n'
        
        # Need to manually append history on first question since using query
        chat_engine.chat_history.append(
            ChatMessage(
                role=MessageRole.USER,
                content = question 
            )
        )
        
        chat_engine.chat_history.append(
            ChatMessage(
            role=MessageRole.ASSISTANT,
            content = response_message.content.strip()
            )
        )
    
    else:
        response = await cl.make_async(chat_engine.stream_chat)(question)

        response_message = cl.Message(content="")

        for token in response.response_gen:
            await response_message.stream_token(token=token)

        msg_content = '\n\nSources:\n'
        title = response.source_nodes[0].metadata["source"]
        link = response.source_nodes[0].metadata["link"]
        msg_content += f'\u00A0 \- [{title}]({link}) '
        page0 = response.source_nodes[0].metadata["page"]
        page1 = response.source_nodes[1].metadata["page"]
        if page0 == page1: msg_content += f'page: {page0}\n'
        else: msg_content += f'pages: {page0} & {page1}\n'
        
    for char in str(msg_content):
        await asyncio.sleep(0.012)
        await response_message.stream_token(token=char)

    QUESTION_COUNT += 1
    
    
    


