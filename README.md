# LlamaIndex_RAG_Memory

<p align="center">
   <strong>Running The UI code for Conversational Memory with RAG that cites sources</strong><br><br>
   <img src="https://github.com/Josh-ee/LlamaIndex_RAG_Memory/assets/74482044/e412ebc3-fe27-499a-b2a7-c1eaa06049a4" alt="ezgif com-optimize-5"/>
</p>

Note: The GIF demo is a real-time representation of the code execution on M2 Max and has not been sped up  


Click [here](https://youtu.be/80ige6hVlE8) for a longer demo

## Introduction

This repository is an example of how to create and use Retrieval Augmented Generation (RAG) with LlamaIndex. 
This is done using open-source models and does not require any API or paid service.  
However, the RAG will work with paid APIs like GPT4-Turbo, just change the llm input in service_context

Here are the libraries used:

   1. Vector Storage -> ChromaDB
   2. Embedding Model -> [BAAI/bge-small-en-v1.5](https://huggingface.co/spaces/mteb/leaderboard) from HuggingFaceEmbeddings
   3. LLM -> [Mistral-11B-OmniMix](https://huggingface.co/TheBloke/Mistral-11B-OmniMix-GGUF) the 4bit quantized GGUF version from TheBloke
   4. This GGUF model is loaded with [LlamaCPP](https://github.com/abetlen/llama-cpp-python.git), which results in solid perforamce
   5. User Interface (UI) -> [Chainlit](https://docs.chainlit.io/integrations/llama-index)



# Vector Database and RAG with LlamaIndex


The `create_VectorDB.ipynb` notebook guides you through the process of creating a vector database using Chroma DB, which stores embeddings from Hugging Face's language models. This vector database is then used by the demo script for RAG.

The `demo_RAG.ipynb` notebook demonstrates how to utilize the created vector database to answer questions based on the documents it contains.


## Part 1: Creating the Vector Database with ChromaDB and Hugging Face Embeddings

Use the `create_vectorDB.ipynb` to create the `RAG_VectorDB`
   1. Download an example PDF from arXiv
   2. Convert the PDF to LlamaIndex Documents
   3. Convert Documents into LlamaIndex Nodes
   4. Create and store the Vector DB


## Part 2: Utilizing the Vector Database with an Open Source LLM Model

Run the `demo_RAG.ipynb` which will step you through 4 different examples:
   1. Load the Foundational LLM and ask a question
   2. Use the LLM with RAG from RAG_VectorDB
   3. Conversational Memory with RAG and Sources

## Performance

This code was developed on a M2 Max with 32GB of RAM. However, you can scale the embedding model and/or the LLM model to better match with your system.  
All of the necessary imports for Mac to utilize MPS are present in the notebooks.  
On an M2 Max, the time from prompt to completed answer is roughly comparable to GPT4 and is more than useable 

