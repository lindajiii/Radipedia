###################################################
# Setup LLM Models and Milvus Database Connection #
# ##################################################

import os
from pymilvus import (connections, MilvusClient, utility)
import streamlit as st
import re


@st.cache_resource
def connect_to_milvus(uri, token):
    con = connections.connect(
      alias='zilliz',
      uri=uri,
      token=token,
    )
    return con

from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings



#embed_model = HuggingFaceEmbeddings(model_name="allenai/biomed_roberta_base")

@st.cache_resource
def load_embedding_model(embedding_model_for_retrieval) :
    embed_model = HuggingFaceEmbeddings(model_name=embedding_model_for_retrieval)
    return embed_model

@st.cache_resource
def load_llm_model(use_longrope, llm_model, llm_model_longrope):
    if use_longrope == False:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        def load_llama_model_quantized(model_id):
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Define quantization configuration for 8-bit or 4-bit
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,                     
                bnb_4bit_compute_dtype=torch.float16
            )

            # Load the model with the specified quantization configuration
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",                    # Automatically allocate layers to GPU/CPU
                torch_dtype=torch.float16,             # Use float16 for reduced memory usage on GPU
                quantization_config=quantization_config,  # Pass the quantization config here
                offload_folder="./VP_storage/offload/" # Folder for offloaded parts if necessary
            )

            return tokenizer, model

        tokenizer, model = load_llama_model_quantized(llm_model) #change as needed

    else :
        from unsloth import FastLanguageModel

        import torch
        max_seq_length = 2048 # We support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

        # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
        fourbit_models = [
            "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
            "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
            "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
            "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
            "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
            "unsloth/Phi-3-medium-4k-instruct",
            "unsloth/gemma-2-9b-bnb-4bit",
            "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

            "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
            "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
            "unsloth/Llama-3.2-3B-bnb-4bit",
            "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        ] # More models at https://huggingface.co/unsloth

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )


        # model = FastLanguageModel.get_peft_model(
        #     model,
        #     r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
        #                       "gate_proj", "up_proj", "down_proj",],
        #     lora_alpha = 16,
        #     lora_dropout = 0, # Supports any, but = 0 is optimized
        #     bias = "none",    # Supports any, but = "none" is optimized
        #     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        #     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        #     random_state = 3407,
        #     use_rslora = False,  # We support rank stabilized LoRA
        #     loftq_config = None, # And LoftQ
        # )

        model = FastLanguageModel.for_inference(model)
    return tokenizer, model

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_milvus import Milvus
from transformers import pipeline

import torch

@st.cache_resource
def initialize_llm_pipeline(_model, _tokenizer, max_new_tokens) :
    hf_pipeline = pipeline("text-generation", model= _model, max_new_tokens=max_new_tokens, device_map="auto", tokenizer = _tokenizer)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm


#@st.cache_resource
def initialize_vectorstore(collection_name, _embed_model, CLUSTER_ENDPOINT, TOKEN):
    vectorstore = Milvus(
        collection_name=collection_name,
        embedding_function = _embed_model, # Placeholder for embedding function
        connection_args={"uri": CLUSTER_ENDPOINT, "token":TOKEN},  # Adjust URI as needed
    )   
    
    return vectorstore


# # Function to format documents
# def format_docs(docs: List[Document]):
#     return "\n\n".join(doc.page_content for doc in docs)


from langchain_core.runnables import RunnableMap, RunnableLambda, RunnablePassthrough

def format_docs(documents):
    """
    Combine text and metadata into two structured formats for the LLM.
    
    Args:
        docs (list of Document): A list of documents, where each document has `text` and `metadata`.
    
    Returns:
        str, str: Formatted context string and two versions of metadata string.
    """


    if isinstance(documents, dict):
        fake_docs= documents['fake_docs']
        docs = documents['retrieved_docs_with_sources']
    else :
        fake_docs = ""
        docs = documents
    
    
    
    metadata_str = ""
    metadata_str_2 = ""  # For the version that combines pages and removes page_num = -1

    seen_documents = {}  # Dictionary to track documents and their page numbers
    
    i = 0
    
    context = ""  # Initialize the context variable

    for doc in docs:
        metadata = doc[0].metadata
        text = doc[0].page_content
        score = doc[1]

        # Extract specific metadata fields
        book_title = metadata.get("book_title", "Unknown Title")
        page_num = metadata.get("page_num", "Unknown Page")
        image_path = metadata.get("image_path", "")

        # Check conditions to decide if the content should be skipped
        if (score <= 0.6 and image_path) or bool(re.search(r'^\d+\.\s', text)) or (i >=10) or bool(re.search(r'\bin less than 50 words\b', text, re.IGNORECASE)) or (text == "[]: Case_type: Case"):
            continue
        else:
            i += 1
            # Adjust the page number if valid
            if page_num != -1: 
                page_num = page_num + 1  # Increment because numbering starts from 0

            # Append metadata and text to the context and metadata strings
            if image_path:  # For image retrieval purposes
                metadata_str += image_path
                metadata_str += f"\n\n {text}\n\n Source: \n {book_title} --Page: {int(page_num)} --Score: {round(score, 4)}  \n\n -- \n\n"
                metadata_str_2 += image_path
                metadata_str_2 += f"\n\n {text}\n\n Image Source: \n {book_title} --Page: {int(page_num)}"
                metadata_str_2 += "\n\n" 
            else:
                metadata_str += f"{text}\n\n Source: \n {book_title} --Page: {int(page_num)} --Score: {round(score, 4)}  \n\n -- \n\n"

            metadata_str += "\n\n"  # Add space between entries

            # Update the context variable by appending valid text
            text_2 = text.replace('\n\n', ' ')
            context += f"{text_2}\n\n"  # Append the text to the context

            # Handle page tracking in seen_documents
            if book_title not in seen_documents:
                seen_documents[book_title] = {"pages": set()}

            # If the page number is valid (not -1), add it to the set of pages
            if page_num == -1:
                seen_documents[book_title]["pages"].add("summary") 
            else:
                seen_documents[book_title]["pages"].add(int(page_num)) # Ensure it's a string

            
        # Construct metadata_str_2 from the seen_documents dictionary
    for book_title, doc_info in seen_documents.items():
        # Ensure all page numbers are converted properly for sorting
        pages = sorted(
            doc_info["pages"],
            key=lambda x: (x == "summary", int(x) if isinstance(x, int) or x.isdigit() else float('inf'))
        )
        # Convert pages to string for joining
        pages_joined = ", ".join(str(p) for p in pages)

        # Append to metadata_str_2
        if image_path:
            metadata_str_2 = metadata_str_2  # Keep as-is (no-op)
        else:
            metadata_str_2 += f"{book_title}\n -- Pages: {pages_joined}"
            metadata_str_2 += "\n\n"  # Add space between entries


    return context, metadata_str, metadata_str_2, fake_docs



def hyde_chain_generation(llm, vectorstore, book_titles, embeddings, k, expr, rag_prompt):
    
    from typing import Optional
    import re
    import numpy as np
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import (
        RunnablePassthrough,
        RunnableLambda,
        Runnable,
        RunnableConfig,
    )
    from langchain_milvus import Milvus
    from langchain_core.runnables.utils import Input, Output


#     fake_doc_prompt = ChatPromptTemplate.from_template(
#         "Generate exactly 3 brief, unique answers to the following question. Each answer must be no longer than 20 words. "
#         "Do not include any extra explanations, symbols, numbers, comments, or notes. Each answer should be clear, concise, and to the point. "
#         "Strictly follow these rules and do not add anything extra. Do not repeat answers. \n\n{query}\n\nAnswers:\n\n"
#     )
    
    
    fake_doc_prompt = ChatPromptTemplate.from_template("Given a radiology related query, generate a paragraph of text, at most 20 words, that answers the query, end with '[END OF ANSWER]'.    Question: \n\n{query}\n\nParagraph:\n\n")
    
#     fake_doc_prompt = ChatPromptTemplate.from_template("""SYSTEM: You are an advanced AI assistant strictly limited to answer user's query. 
#     Given a radiology related query, generate a short paragraph of text that answers the query, at most 20 words. 
    
#     USER: 
#     <query>
#     {query}
#     </query>
    
#     ASSISTANT:""")


    fake_doc_chain = (
        {"query": RunnablePassthrough()} | fake_doc_prompt | llm | StrOutputParser()
    )


    class HydeRetriever(Runnable):
        def __init__(self, vectorstore):
            self.vectorstore = vectorstore
            self.hyde_retriever = {
                "fake_generation": fake_doc_chain,
                "query": RunnablePassthrough(),
            } | RunnableLambda(self._retrieve_from_fake_docs)

        @classmethod
        def from_vectorstore(cls, vectorstore: Milvus):
            return cls(vectorstore=vectorstore)

        def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
            return self.hyde_retriever.invoke(input)

        def _retrieve_from_fake_docs(self, _dict):
            fake_generation = _dict["fake_generation"]
            query = _dict["query"]

            # Get the answers part of the text
            #answers_text = fake_generation.split("Answers:")[1].strip()
            answers_text = fake_generation.split("Paragraph:")[1].split("[END OF ANSWER]")[0].strip()
            #answers_text = fake_generation.split("ASSISTANT:")[1].strip()
            

            # Create a clean list of answers without numbering
            fake_docs = [re.sub(r'^\d+\.\s*', '', line.strip()) for line in answers_text.split('\n') if line.strip()]
            
            # Store fake_docs for further use
            _dict["fake_docs"] = fake_docs

            # Concatenate the document embeddings
#             doc_vectors = embeddings.embed_documents(fake_docs)
#             vector_array = np.array(doc_vectors)
            
#             query_vector = embeddings.embed_query(query)
#             query_vector = np.array(query_vector)

#             vector_array = np.concatenate(([query_vector], vector_array), axis=0)

            combined_text = query + " " + " ".join(fake_docs)

            # Embed the combined text
            vector_array = embeddings.embed_query(combined_text)  # Single embedding


            # Search average embedding
            #average_doc_vector = np.mean(vector_array, axis=0).tolist()
            res = self.vectorstore.similarity_search_with_score_by_vector(embedding=vector_array, k=k, expr=expr)
        

            # Add fake_docs to the response
            _dict["retrieved_docs_with_sources"] = res

            return _dict

    hyde_retriever = HydeRetriever.from_vectorstore(vectorstore)

    # Chain with the new process to include fake_docs in the final output
    hyde_chain = (
        RunnableMap(
            {
                "context": hyde_retriever | format_docs,  # Retrieve and format (returns both docs and metadata)
                "question": RunnablePassthrough(),  # Pass question
            }
        )
        | RunnableLambda(
            lambda inputs: {
                "formatted_docs": inputs["context"][0],  # Capture the first output (documents)
                "metadata": inputs["context"][1],  # Capture the second output (metadata)
                "metadata2": inputs["context"][2], 
                "final_prompt": rag_prompt.invoke({"context": inputs["context"][0], "question": inputs["question"]}),
                "fake_docs": inputs["context"][3],  # Add the fake_docs here
            }
        )
        | RunnableLambda(
            lambda inputs: {
                "retrieved_docs": inputs["formatted_docs"],  # Use the formatted docs
                "metadata": inputs["metadata"],  # Pass metadata along
                "metadata2": inputs["metadata2"], 
                "fake_docs": inputs["fake_docs"],  # Pass fake_docs along
                #"synthesized_answer": llm.invoke(inputs["final_prompt"]).replace("[END OF ANSWER]", "").strip(),  # Generate the answer using the final prompt
                "synthesized_answer": re.sub(r'\[END OF ANSWER\].*', '', llm.invoke(inputs["final_prompt"])).strip()
                
            }
        )
        | RunnableLambda(
            lambda inputs: {
                # Ensure that the metadata is immediately appended after each retrieved_doc
                "retrieved_docs": inputs["retrieved_docs"],
                #"retrieved_docs_with_sources": f"{inputs['retrieved_docs']} \n\nSources:\n\n {''.join(inputs['metadata'])} \n\n Fake Docs Generated:\n\n {', '.join(inputs['fake_docs'])}",
                "retrieved_docs_with_sources":f"**Hypothetical Answer Generated by HyDE:**\n\n {', '.join(inputs['fake_docs'])}  \n\n ------ \n\n **Retrieved Documents:**\n\n {''.join(inputs['metadata'])}  \n\n ------ \n\n ",
                "answer_with_sources": f"{inputs['synthesized_answer']} \n\nText Sources:\n\n {''.join(inputs['metadata2'])} \n\n--\n\n",  # For the final answer
            }
        )
    )

    return fake_doc_chain, hyde_chain, hyde_retriever



def vanilla_chain_generation(llm, vectorstore, book_titles, k, expr,rag_prompt):
    
    
    class CustomRetrieverWithScore:
        def __init__(self, vectorstore, k=k, expr=None):
            self.vectorstore = vectorstore
            self.k = k
            self.expr = expr

        def get_relevant_documents(self, query):
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k, expr=self.expr)
            
            
            return docs_with_scores
    
    
    retriever = CustomRetrieverWithScore(vectorstore, k=k, expr=expr)
    

    retriever_runnable= RunnableLambda(
        lambda query: retriever.get_relevant_documents(query)
    )
    
    
    vanilla_rag_chain = (
        # Retrieve documents and format them
        RunnableMap({"context": retriever_runnable | format_docs, "question": RunnablePassthrough()})  # Pass documents and question
        | RunnableLambda(
            lambda inputs: {
                "formatted_docs": inputs["context"][0],  # Capture the first output (documents)
                "metadata": inputs["context"][1],  # Capture the second output (metadata)
                "metadata2": inputs["context"][2], 
                "final_prompt": rag_prompt.invoke({"context": inputs["context"][0], "question": inputs["question"]}),
            }
        )
        | RunnableLambda(
            lambda inputs: {
                "retrieved_docs": inputs["formatted_docs"],  # Use the formatted docs
                "metadata": inputs["metadata"],  # Pass metadata along
                "metadata2": inputs["metadata2"], 
               # "synthesized_answer": llm.invoke(inputs["final_prompt"]).replace("[END OF ANSWER]", "").strip()  # Generate the answer using the final prompt
                "synthesized_answer": re.sub(r'\[END OF ANSWER\].*', '', llm.invoke(inputs["final_prompt"])).strip()
            }
        )
        | RunnableLambda(
            lambda inputs: {
            # Ensure that the metadata is immediately appended after each retrieved_doc
            "retrieved_docs": inputs["retrieved_docs"],
            #"retrieved_docs_with_sources":f"{inputs['retrieved_docs']} \n\nSources:\n\n {''.join(inputs['metadata'])}",
            "retrieved_docs_with_sources":f"**Retrieved Documents:**\n\n {''.join(inputs['metadata'])}  \n\n ------ \n\n ",
            "answer_with_sources": f"{inputs['synthesized_answer']} \n\nText Sources:\n\n {''.join(inputs['metadata2'])}"  # For the final answer
        }
    )
    )
        
    return vanilla_rag_chain

def initialize_chain(use_hyde, llm, vectorstore, book_titles, embeddings, k, expr, rag_prompt):
    k = 25 #generate many first then choose top 10 (because many redundant chunk to filter)
    if use_hyde == True :
        fake_doc_chain, chain, hyde_retriever = hyde_chain_generation(llm, vectorstore, book_titles, embeddings, k, expr,rag_prompt)
    else :
        chain = vanilla_chain_generation(llm, vectorstore, book_titles, k, expr,rag_prompt, config)

    return chain


def rag_and_synthesize(query, _chain, use_longrope, use_hyde, config):
    if use_longrope == True :
        print("Using longrope")
    else :
        print("Without using longrope")
    

    
    result = _chain.invoke(query, config = config)
    if use_hyde == True :
        print("Using HyDE")
    else :
        print("Without using HyDE")
        
    print("Retrieved docs:\n\n" ,result['retrieved_docs_with_sources'])
#        print("Retrieved docs:\n\n" ,hyde_result)
   # match = re.search(r"Answer:\s*(.*)", result['answer_with_sources'], re.DOTALL)  # Match in multiline string
    match = re.search(r"ASSISTANT:\s*(.*)", result['answer_with_sources'], re.DOTALL)  # Match in multiline string

#        match = re.search(r"Answer:\s*(.*)", hyde_result, re.DOTALL)  # Match in multiline string
    result_final = match.group(1).strip() if match else ""  # Extract matched group
    print("Final Answer:\n\n", result_final)
    return result['retrieved_docs'],result['retrieved_docs_with_sources'], result_final



def retrieve_image(query, vectorstore_fig, k_figure, expr_fig):
    # Define custom retriever
    class CustomRetrieverWithScore:
        def __init__(self, vectorstore_fig, k_figure, expr_fig):
            self.vectorstore = vectorstore_fig
            self.k = k_figure
            self.expr = expr_fig

        def get_relevant_documents(self, query):
            """
            Perform similarity search with scores and return results.
            """
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k, expr=self.expr)
            return docs_with_scores

    # Instantiate custom retriever
    
    img_retriever = CustomRetrieverWithScore(vectorstore_fig, k_figure, expr_fig)

    img_retriever_runnable = RunnableLambda(
        lambda query: img_retriever.get_relevant_documents(query)
    )

    # Define chain
    vanilla_rag_chain = (
        RunnableMap({"context": img_retriever_runnable | format_docs, "question": RunnablePassthrough()})
         | RunnableLambda(
            lambda inputs: {
                "result": inputs["context"][1],
                "result_final":  inputs["context"][2]
            }
        )
        | RunnableLambda(       # Combines formatted docs and metadata
            lambda inputs: {
                "retrieved_docs_with_sources": f"**Retrieved Images:**\n\n{''.join(inputs['result'])}\n\n------\n\n",
                "final_image_to_show": f"{''.join(inputs['result_final'])}",
            }
        )
    )


    # Run the chain
    result = vanilla_rag_chain.invoke(query)
    return result["retrieved_docs_with_sources"], result["final_image_to_show"]


def opik_config():
    from opik.integrations.langchain import OpikTracer
    opik_tracer = OpikTracer()

    config = {
        'callbacks' : [opik_tracer]
    }
    return config
