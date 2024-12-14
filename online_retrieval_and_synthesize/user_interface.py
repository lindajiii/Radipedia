#######################
# Setup the Variables #
#######################

embedding_model_for_retrieval = 'sentence-transformers/all-MiniLM-L6-v2' #using huggingface embedding function -- ensure that it is the same as the one used to push to milvus

llm_model = "meta-llama/Llama-3.2-3B-Instruct" #will be loaded with quantized framework, alternative : "aaditya/OpenBioLLM-Llama3-8B" or other methods 
llm_model_longrope = "unsloth/Llama-3.2-3B-Instruct" #ideally same model type as the llm_model for fair comparison -- I assume the one from unsloth support longrope? so I specify differently


TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
CLUSTER_ENDPOINT = "https://xxxxxxxxxx.serverless.gcp-us-west1.cloud.zilliz.com"

collection_name = "radiology_paper_raptor" #for default
collection_figure_name = "radiology_paper_figures" #for default
image_directory = "/home/jupyter/project/VP_storage/output_papers/" #for default

#book_titles = ["Imaging evaluation of the liver in oncology patients"] #or "all" for searching in all documents, or ["a", "b"] for more than 1 document
document = "all"

max_new_tokens=200


#we make these dynamic below
# use_hyde = True
# use_longrope = True
# use_raptor = True

k = 40 #how many vector to retrieve #actually this is not used now, we have a mechanism to eventually choose the top 10 for now.
k_figure=2 




# PROMPT_TEMPLATE = """
# You are an advanced AI assistant designed to provide concise, fact-based, and statistically accurate answers to questions. 
# When answering, prioritize clarity, precision, and the use of verifiable data. 

# Use the provided context enclosed in `<context>` tags to generate your response. If the context does not contain enough information to answer the question, state: "I don't have enough information to answer this question." 
# Strictly do not attempt to use your prior knowledge, fabricate, or infer answers beyond the provided context. Do not give your personal opinion or comment.

# <context>
# {context}
# </context>

# <question>
# {question}
# </question>

# Your response must:
# 1. Be specific and directly address the question.
# 2. Include statistics, numbers, or other quantifiable details whenever possible.
# 3. Avoid verbosity, elaboration, and unnecessary phrases like "note" or "provided context" or "I think".
# 4. Use at most 50 words.
# 5. Do not need to close with </answer>.

# Answer:"""

PROMPT_TEMPLATE = """SYSTEM: You are an advanced AI assistant strictly limited to providing concise, fact-based, and statistically accurate answers to questions.
End your generated answer with "[END OF ANSWER]".

You must:
1. Rely exclusively on the provided context enclosed in `<context>` tags.
2. Never use prior knowledge, fabricate, infer, or offer opinions.
3. Ensure all statements are verifiable within the context.
4. Keep answers specific, concise,  and at most 50 words.
5. Prioritize clarity and quantifiable details (e.g., numbers, statistics).

Failure to adhere or lack of confidence will result in outputting: "I'm sorry I don't have enough information to answer this question. [END OF ANSWER]"

USER:
<context>
{context}
</context>

<question>
{question}
</question>

ASSISTANT:"""





from langchain_core.prompts import PromptTemplate



rag_prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

#from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import pandas as pd
from streamlit_float import *
import time
import os

from resource_initialization import connect_to_milvus,load_embedding_model,load_llm_model
from resource_initialization import initialize_llm_pipeline,initialize_vectorstore
from resource_initialization import format_docs,hyde_chain_generation,vanilla_chain_generation
from resource_initialization import initialize_chain,rag_and_synthesize, retrieve_image
from resource_initialization import opik_config



st.set_page_config(
    page_title="Radipedia - UChicago Medicine",
    layout="wide",  # Wider layout
    initial_sidebar_state="collapsed",
)

#checkboxes to add flexibility to explore around

if "show_retriever" not in st.session_state:
    st.session_state.show_retriever = True
if "use_raptor" not in st.session_state:
    st.session_state.use_raptor = True 
if "use_hyde" not in st.session_state:
    st.session_state.use_hyde = True 
if "use_longrope" not in st.session_state:
    st.session_state.use_longrope = True 

# Sidebar Checkbox for Toggling Retriever Visibility
show_retriever = st.sidebar.checkbox(
    "Show Retrieval", value=st.session_state.show_retriever
)
use_raptor = st.sidebar.checkbox(
    "Use RAPTOR", value=st.session_state.use_raptor
)
use_hyde = st.sidebar.checkbox(
    "Use HyDE", value=st.session_state.use_hyde
)
use_longrope = st.sidebar.checkbox(
    "Use LongRoPE", value=st.session_state.use_longrope
)

# Update session state based on the checkbox
if show_retriever != st.session_state.show_retriever:
    st.session_state.show_retriever = show_retriever
if use_raptor != st.session_state.use_raptor:
    st.session_state.use_raptor = use_raptor
if use_hyde != st.session_state.use_hyde:
    st.session_state.use_hyde = use_hyde
if use_longrope != st.session_state.use_longrope:
    st.session_state.use_longrope = use_longrope



# Adjust column layout dynamically
if st.session_state.show_retriever:
    col1, col2 = st.columns([2, 1])  # Normal chatbot and retriever width
else:
    col1 = st.columns([1])[0]  # Full width for chatbot when retriever is hidden

    




if "connect_to_milvus" not in st.session_state:
    st.session_state["connect_to_milvus"] = connect_to_milvus(CLUSTER_ENDPOINT, TOKEN)
if "embed_model" not in st.session_state:
    st.session_state["embed_model"] = load_embedding_model(embedding_model_for_retrieval)
if ("tokenizer" not in st.session_state) & ("model" not in st.session_state):
    st.session_state["tokenizer"], st.session_state["model"]= load_llm_model(use_longrope, llm_model, llm_model_longrope)
if "llm" not in st.session_state:
    st.session_state["llm"] = initialize_llm_pipeline(st.session_state["model"], st.session_state["tokenizer"], max_new_tokens)
if "collection_name" not in st.session_state:
    st.session_state["collection_name"] = collection_name
    st.session_state["collection_figure_name"] = collection_figure_name
    st.session_state["image_directory"] = image_directory
if "vectorstore" not in st.session_state:    
    st.session_state["vectorstore"] = initialize_vectorstore(st.session_state["collection_name"], st.session_state["embed_model"], CLUSTER_ENDPOINT, TOKEN)
    st.session_state["vectorstore_fig"] = initialize_vectorstore(st.session_state["collection_figure_name"], st.session_state["embed_model"], CLUSTER_ENDPOINT, TOKEN)
if "document" not in st.session_state:
    st.session_state["document"]="Select All"
if "expr" not in st.session_state:
    if st.session_state["use_raptor"] == True :
        st.session_state["expr"]=""
    else : 
        st.session_state["expr"]= "level == 0"
if "expr_fig" not in st.session_state:
    st.session_state["expr_fig"]=""

    


    

import re
from PIL import Image


# Initialize float with theme
#float_init(theme=True, include_unstable_primary=False)




# App title
st.title("Radipedia - UChicago Medicine")




# Initialize an empty dictionary to store the cleaned file names
reference_list = {"Books":['Select All','gastrointestinal-imaging-the-requisites-fourth-edition_mainpages',
       'CT_and_MRI_of_the_Whole_Body',
       'Liver_imaging_MRI_with_CT_correlation',
       'Mayo_Clinic_Gastrointestinal_Imaging_Review',
       'Radiology_Illustrated_Hepatobiliary_and_Pancreatic_Radiology'],"Papers":['Select All','Imaging evaluation of the liver in oncology patients',
       'Hyperintense Liver Masses at Hepatobiliary Phase Gadoxetic Acidenhanced MRI',
       'How to Use LI-RADS to Report Liver CT and MRI Observations',
       'Abbreviated MRI for Hepatocellular Carcinoma Screening and Surveillance',
       'Role of MRI in Evaluation of Spectrum of Liver Lesions in Cirrhotic Patients'], "System Guides":['Select All','Reporting-Reporting_and_Data_Systems_Support',
       'Getting_Started-Reporting_and_Data_Systems_Support',
       'Imaging_Features-Reporting_and_Data_Systems_Support',
       'Management-Reporting_and_Data_Systems_Support',
       'Treatment_Response-Reporting_and_Data_Systems_Support',
       'Diagnostic_Categories-Reporting_and_Data_Systems_Support',
       "What's_new_in_v2018-Reporting_and_Data_Systems_Support",
       'Technique-Reporting_and_Data_Systems_Support',
       'Application-Reporting_and_Data_Systems_Support',
       'Diagnosis-Reporting_and_Data_Systems_Support',
       'LI-RADS_US_Surveillance_v2024_Core',
       'LI-RADS_CTMR_Radiation_TRA_v2024_Core',
       'LI-RADS_CTMR_Nonradiation_TRA_v2024_Core', 'LI-RADS_2018_Core']}


list_all_text_references = ['Imaging evaluation of the liver in oncology patients',
       'Hyperintense Liver Masses at Hepatobiliary Phase Gadoxetic Acidenhanced MRI',
       'How to Use LI-RADS to Report Liver CT and MRI Observations',
       'Abbreviated MRI for Hepatocellular Carcinoma Screening and Surveillance',
       'Role of MRI in Evaluation of Spectrum of Liver Lesions in Cirrhotic Patients', 'gastrointestinal-imaging-the-requisites-fourth-edition_mainpages',
       'CT_and_MRI_of_the_Whole_Body',
       'Liver_imaging_MRI_with_CT_correlation',
       'Mayo_Clinic_Gastrointestinal_Imaging_Review',
       'Radiology_Illustrated_Hepatobiliary_and_Pancreatic_Radiology', 'Reporting-Reporting_and_Data_Systems_Support',
       'Getting_Started-Reporting_and_Data_Systems_Support',
       'Imaging_Features-Reporting_and_Data_Systems_Support',
       'Management-Reporting_and_Data_Systems_Support',
       'Treatment_Response-Reporting_and_Data_Systems_Support',
       'Diagnostic_Categories-Reporting_and_Data_Systems_Support',
       "What's_new_in_v2018-Reporting_and_Data_Systems_Support",
       'Technique-Reporting_and_Data_Systems_Support',
       'Diagnosis-Reporting_and_Data_Systems_Support',
       'LI-RADS_US_Surveillance_v2024_Core',
       'LI-RADS_CTMR_Radiation_TRA_v2024_Core',
       'LI-RADS_CTMR_Nonradiation_TRA_v2024_Core', 'LI-RADS_2018_Core']

list_all_figure_references = ['WJH-13-1936_mainpages',
       'fujita-et-al-2019-hyperintense-liver-masses-at-hepatobiliary-phase-gadoxetic-acid-enhanced-mri-imaging-appearances-and_mainpages',
       'm-cunha-et-al-2021-how-to-use-li-rads-to-report-liver-ct-and-mri-observations_mainpages',
       'an-et-al-2020-abbreviated-mri-for-hepatocellular-carcinoma-screening-and-surveillance_mainpages',
       'RoleofMRIinEvaluationofSpectrumofLiverLesionsinCirrhoticPatients-JAPI_mainpages',
        'gastrointestinal-imaging-the-requisites-fourth-edition_mainpages',
       'CT and MRI of the Whole Body, 2-Volume Set, 6e, Volume I_mainpages',
       'Liver imaging _ MRI with CT correlation_mainpages','Mayo Clinic Gastrointenstinal Imaging Review_mainpages'
       'Radiology Illustrated_ Hepatobiliary and Pancreatic Radiology_mainpages', 
       'Reporting _ RADS - Reporting and Data Systems Support_mainpages',
       'Getting Started _ RADS - Reporting and Data Systems Support_mainpages',
       'Imaging Features _ RADS - Reporting and Data Systems Support_mainpages',
       'Management _ RADS - Reporting and Data Systems Support_mainpages',
       'Treatment Response _ RADS - Reporting and Data Systems Support_mainpages',
       'Diagnostic Categories _ RADS - Reporting and Data Systems Support_mainpages',
       "What's New in v2018 _ RADS - Reporting and Data Systems Support_mainpages",
       'Technique _ RADS - Reporting and Data Systems Support_mainpages',
       'Diagnosis _ RADS - Reporting and Data Systems Support_mainpages',
       'LI-RADS US Surveillance v2024 Core_mainpages',
       'LI-RADS CTMR Radiation TRA v2024 Core_mainpages',
       'LI-RADS CTMR Nonradiation TRA v2024 Core_mainpages',
       'LI-RADS 2018 Core_mainpages']


mapping = dict(zip(list_all_text_references , list_all_figure_references))

if __name__ == "__main__":
    # adding "select" as the first and default choice
    st.session_state["reference_type"] = st.selectbox('Select Reference Type (Default : Paper)', options=['Select Reference']+list(reference_list.keys()))
    # display selectbox 2 if manufacturer is not "select"
    if st.session_state["reference_type"]  != 'Select Reference':
        st.session_state["document"] = st.selectbox('Select Document to Refer', options=reference_list[st.session_state["reference_type"]])
        
    if st.session_state["reference_type"] == "Books":
        st.session_state["collection_name"] = "radiology_book_raptor"
        st.session_state["collection_figure_name"] = "radiology_book_figures"
        st.session_state["image_directory"] = "/home/jupyter/project/VP_storage/output_books/"
    elif st.session_state["reference_type"] == "Papers":
        st.session_state["collection_name"] = "radiology_paper_raptor"
        st.session_state["collection_figure_name"] = "radiology_paper_figures" 
        st.session_state["image_directory"] = "/home/jupyter/project/VP_storage/output_papers/"
    elif st.session_state["reference_type"] == "System Guides":
        st.session_state["collection_name"] = "radiology_system_guide_raptor"
        st.session_state["collection_figure_name"] =  "radiology_system_guide_figures"
        st.session_state["image_directory"] = "/home/jupyter/project/VP_storage/output_system_guides/"
        
    
#     document =  st.session_state["document"]
#     print(document)
#     if st.button('Submit'):
#         if st.session_state["document"] == "Select All":
#             st.session_state["expr"] = ""
#             if use_raptor == False :
#                 st.session_state["expr"]  = "level == 0"
#         else: 
#             st.session_state["expr"] ="book_title in [" + ", ".join([f"'{title}'" for title in [document]]) + "]"
#             if use_raptor == False :
#                 st.session_state["expr"] = st.session_state["expr"] + "&&  (level == 0)"
        
    
#     if st.button('Submit'):
#         if st.session_state["document"] == "Select All":
#             st.session_state["expr"] = ""
#             st.session_state["expr_fig"] = ""
#             if use_raptor == False:
#                 st.session_state["expr"] = "level == 0"
#         else:
#             # Mengambil versi List B berdasarkan input di List A
#             document = st.session_state["document"]
#             mapped_document = mapping.get(document, document)  # Default jika tidak ditemukan

#             st.session_state["expr"] = "book_title in [" + ", ".join([f"'{title}'" for title in [document]]) + "]"
#             st.session_state["expr_fig"] = "book_title in [" + ", ".join([f"'{title}'" for title in [mapped_document]]) + "]"

#             if use_raptor == False:
#                 st.session_state["expr"] += " && (level == 0)"



    if st.button('Submit'):
        if st.session_state["document"] != "Select All":
            document = st.session_state["document"]
            mapped_document = mapping.get(document, document)  
            image_directory = st.session_state["image_directory"]
        st.session_state["vectorstore"] = initialize_vectorstore(st.session_state["collection_name"], st.session_state["embed_model"], CLUSTER_ENDPOINT, TOKEN)
        st.session_state["vectorstore_figure"] = initialize_vectorstore(st.session_state["collection_figure_name"], st.session_state["embed_model"], CLUSTER_ENDPOINT, TOKEN)
        st.write('You selected: ' + st.session_state["document"] + " (" + st.session_state["reference_type"] +")")
        
    if st.session_state["document"] == "Select All":
        st.session_state["expr"] = ""
        st.session_state["expr_fig"] = ""
        if st.session_state["use_raptor"] == False:
            st.session_state["expr"] = "level == 0"
    else:
        document = st.session_state["document"]
        mapped_document = mapping.get(document, document)  
        image_directory = st.session_state["image_directory"]

        st.session_state["expr"] = "book_title in [" + ", ".join([f"'{title}'" for title in [document]]) + "]"
        st.session_state["expr_fig"] = "book_title in [" + ", ".join([f"'{title}'" for title in [mapped_document]]) + "]"

        if st.session_state["use_raptor"] == False:
            st.session_state["expr"] += " && (level == 0)"


        # st.write(st.session_state["collection_name"])

        
#         st.write(st.session_state["document"])
#         st.write(st.session_state["expr"])
        
        

        

# Regex to detect image paths
image_path_pattern = r"(?:\./|/)?[\w\-.\/]+(?:\.jpg|\.png|\.jpeg)"

# Function to process text and maintain order of images and text
def display_images_and_text(text, main_folder):
    # Split the text into parts (text and image paths)
    parts = re.split(f"({image_path_pattern})", text)

    for part in parts:
        if re.match(image_path_pattern, part):  # If it's an image path
            # Build the full path
            full_path = os.path.join(main_folder, part.lstrip("./"))
            try:
                # Display the image
                st.image(Image.open(full_path), caption=f"Image from: {full_path}", use_container_width=True)
            except FileNotFoundError:
                st.error(f"Image file not found: {full_path}")
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            # Display text
            st.write(part)       
    return ""
    




col1, col2 = st.columns([3, 2])

# Initialize session state for messages, bot response, feedback status
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "bot_response" not in st.session_state:
    st.session_state["bot_response"] = ""
if "bot_response_image" not in st.session_state:
    st.session_state["bot_response_image"] = ""
if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = ""
if "chain" not in st.session_state:
    st.session_state["chain"] = []
if "retrieval_result_final" not in st.session_state:
    st.session_state["retrieval_result_final"] = ""
if "retrieval_result_fig" not in st.session_state:
    st.session_state["retrieval_result_fig"] = ""
if "final_answer" not in st.session_state:
    st.session_state["final_answer"] = ""
if "final_answer_image" not in st.session_state:
    st.session_state["final_answer_image"] = ""
if "fake_docs" not in st.session_state:
    st.session_state["fake_docs"] = ""
if "feedback_given" not in st.session_state:
    st.session_state["feedback_given"] = False  # Track if feedback was given
if "retrieval_duration" not in st.session_state:
    st.session_state["retrieval_duration"] = ""
if "synthesis_duration" not in st.session_state:
    st.session_state["synthesis_duration"] = 0.0
    

# Function to save message and feedback to CSV
def save_to_csv(user_message, assistant_response, retrieval, feedback, reason=""):
    data = {
        "user_message": [user_message],
        "assistant_response": [assistant_response],
        "retrieval": [retrieval],
        "feedback": [feedback],
        "reason":[reason]
    }
    df = pd.DataFrame(data)
    
    # Append to CSV or create it if it doesn't exist
    if not os.path.exists("feedback_log.csv"):
        df.to_csv("feedback_log.csv", index=False)
    else:
        df.to_csv("feedback_log.csv", mode='a', header=False, index=False)

if 'clicked_button_like' not in st.session_state:
    st.session_state.clicked_button_like = False
    
if  'clicked_button_dislike'  not in st.session_state:
    st.session_state.clicked_button_dislike = False
    
    
if 'reason' not in st.session_state:
    st.session_state.reason = ""

def click_button_like():
    st.session_state.clicked_button_like = True
    st.session_state.clicked_button_dislike = False
    save_to_csv(st.session_state["user_prompt"], st.session_state["bot_response"], st.session_state["retrieval_result_final"],feedback="Like")
    st.success("Thank you for your feedback!")
    

def click_button_dislike():
    st.session_state.clicked_button_dislike = True
    st.session_state.clicked_button_like = False

#    save_to_csv(st.session_state["user_prompt"], st.session_state["bot_response"], st.session_state["retrieval_result_final"],feedback="Dislike")
    
# Function to save Dislike feedback with reasoning
def submit_dislike_reason(reason):
    save_to_csv(
        st.session_state["user_prompt"],
        st.session_state["bot_response"],
        st.session_state["retrieval_result_final"],
        feedback="Dislike",
        reason=reason
    )


# Layout for Chatbot and Retrieval
with col1:
    st.subheader("Chatbot")
    with st.container(height = 1000):

        
        
        


        # Display previous messages in chat
        # for message in st.session_state.messages:
        #     if message["role"]=="user":
        #         with st.chat_message(message["role"]):
        #             st.markdown(message["content"])
        #     else:
        #         with st.chat_message(message["role"]):
        #             st.markdown(message["content"])
        #             st.markdown(display_images_and_text(message["image"], message["image_directory"]))
        for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["image"]!="" :
                        st.markdown(display_images_and_text(message["image"], message["image_directory"]))


        # Container to display chat messages
        messages_cont = st.container()
        
        if st.session_state.clicked_button_dislike:
            with st.container():  # Wrap in a container to manage UI components
                reason = st.text_area(
                    "Please provide your reasoning:",
                    value=st.session_state.reason,
                    placeholder="Write your reasoning here..."
                )
                if st.button("Submit Reasoning"):
                    submit_dislike_reason(reason)
                    #st.success("Thank you for your feedback!")
                    # Reset state to hide the text area
                    st.session_state.clicked_button_dislike = False
                    st.session_state.reason = ""




        # User input for chat
        if prompt := st.chat_input("Ask me a question"):
            # Check if there's an unanswered feedback for previous bot response
            if st.session_state["bot_response"] and st.session_state.clicked_button_like == False and st.session_state.clicked_button_dislike == False:
                save_to_csv(st.session_state["user_prompt"], st.session_state["bot_response"], st.session_state["retrieval_result_final"] ,feedback="No feedback")

            # Reset feedback tracking for the new response
            st.session_state["feedback_given"] = False

            # Store new prompt in session state
            st.session_state.messages.append({"role": "user", "content": prompt, "image":"", "image_directory":""})
            st.session_state["user_prompt"] = prompt

            # Display user input in chat
            with messages_cont.chat_message(name="user"):
                st.markdown(prompt)
                

    
           
            


                
# with col2:
#     st.subheader("Retrieval")
#     with st.container(height = 1000):
if prompt : 
    start_retrieval_time = time.time()
    st.session_state["chain"] = initialize_chain(st.session_state["use_hyde"], st.session_state["llm"], st.session_state["vectorstore"],st.session_state["document"],st.session_state["embed_model"], k,st.session_state["expr"],rag_prompt)
#    st.markdown(st.session_state["expr"]) -- is moved to bottom part of code
    retrieved_docs,st.session_state["retrieval_result_final"],st.session_state["final_answer"] =  rag_and_synthesize(st.session_state["user_prompt"], st.session_state["chain"], st.session_state["use_longrope"], st.session_state["use_hyde"], opik_config())
    st.session_state["retrieval_result_fig"],st.session_state["final_answer_image"]  =  retrieve_image(st.session_state["user_prompt"],st.session_state["vectorstore_fig"], k_figure, st.session_state["expr_fig"])
    ret_duration = time.time() - start_retrieval_time
    st.session_state["retrieval_duration"] = f"**Retrieval Duration**: {ret_duration:.2f} seconds"

            
if st.session_state.show_retriever:                
    with col2:
        st.subheader("Retrieval")
        with st.container(height = 1000):
            #st.markdown(st.session_state["expr"])
            st.markdown(st.session_state["retrieval_result_final"])
            display_images_and_text(st.session_state["retrieval_result_fig"],image_directory)
            st.markdown(st.session_state['retrieval_duration'])

if st.session_state["retrieval_result_final"] != "":
    with messages_cont.chat_message("assistant"):

        st.session_state["retrieval_result_final"] = ""



        # Display assistant's response

        st.markdown(st.session_state["final_answer"])


        # Save response to session state

        st.session_state["bot_response"] = st.session_state["final_answer"] 
        display_images_and_text(st.session_state["final_answer_image"], image_directory)
       # display_images_and_text(st.session_state["retrieval_result_fig"], st.session_state["image_directory"])


        st.session_state.messages.append({"role": "assistant", "content": st.session_state["final_answer"], "image":st.session_state["final_answer_image"], "image_directory":image_directory})


        # Create columns for Like and Dislike buttons
        like_col, dislike_col = st.columns(2)
        with like_col:
            st.session_state.clicked_button_like = False
            st.button("👍 Like", on_click = click_button_like)

        with dislike_col:
            st.session_state.clicked_button_dislike = False
            st.button("👎 Dislike", on_click = click_button_dislike)




            
                
st.markdown("---")
st.caption("Powered by Team 6 - LLM in Healthcare - MS in Applied Data Science University of Chicago")
