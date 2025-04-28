import streamlit as st
from src.helper import download_hugging_face_embeddings, predict_xray
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *              
from PIL import Image       
import torch, torch.nn as nn, torchxrayvision as xrv
import os
import time
st.set_page_config(layout="wide")

########################################################
########################################################
########################################################
# MESSAGE WHILE CODE RUNS
placeholder = st.empty()
placeholder.info( "WE ARE GETTING OUR MODELS READY 🚀")


########################################################
########################################################
########################################################

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# INITIALIZE EMBEDDINGS
embeddings = download_hugging_face_embeddings()

# LOAD OUR EMBEDDINGS FROM PINECONE AND INSTANTIATE RETRIEVER
index_name = "designing-ai-medical-chatbot"
# We can now load the knowledge base into an object so we can perform queries
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
# Instantiate a retriever
retriever = docsearch.as_retriever(search_type='similarity', # We will be searching using similarity
                                   search_kwargs={"k":3}) # The retriever will return 3 results


# CREATE INSTANCE OF OPEN AI LLM MODEL
llm = OpenAI(
    temperature=0.4, # This controls the model's creativity/randomness. 0.4 makes sure it is more focused on predictable answers
    max_tokens=500) # Limits the maximum length of the model's output


# CREATE THE TEMPLATE FOR THE PROMPT TO THE LLM
# the prompt is what you send to the LLM (language model).
# It builds the message format that the LLM needs to understand what to do.
# It includes the system_prompt from the src.prompt file and the prompt for the user

prompt_to_llm = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


# The question_answer_chain is responsible for generating an answer based on provided context documents and a user question. 
# It uses the create_stuff_documents_chain function, which means it takes all retrieved documents, combines ("stuffs") them into a single prompt template, 
# and feeds this structured prompt to the language model (LLM). 
# The system prompt defines the behavior (e.g., answer concisely based on context), and the human input is the actual user question. 
# This chain does not handle document retrieval itself — it purely focuses on preparing the input to the LLM and generating a coherent, context-based answer. 
# Its role is to ensure that the model answers using only the given context, while maintaining the tone and constraints set by the system instructions.
question_answer_chain = create_stuff_documents_chain(llm, prompt_to_llm)


# The rag_chain is the full Retrieval-Augmented Generation (RAG) pipeline that combines two essential steps: 
# retrieving relevant documents and generating an answer. 
# It first uses a retriever (such as a vector store search) to find the most relevant context documents based on the user's input question. 
# These retrieved documents are then passed to the question_answer_chain, which constructs the final prompt and calls the language model to produce a concise, accurate answer. 
# By linking retrieval and generation into a single chain, rag_chain automates the process of sourcing information and reasoning over it, 
# making it powerful for tasks where dynamic, knowledge-grounded responses are needed.

rag_chain = create_retrieval_chain(retriever, question_answer_chain)


################ LOAD COMPUTER VISION MODEL
@st.cache_resource
def load_my_model():
    # 1) Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2) Rebuild CheXNet→binary
    model = xrv.models.DenseNet(weights='densenet121-res224-all')
    orig_head = model.classifier
    in_feat = orig_head.in_features if hasattr(orig_head,'in_features') else orig_head[0].in_features
    model.classifier = nn.Linear(in_feat, 1)

    def forward_no_opnorm(x):
        feats = model.features(x)
        return model.classifier(feats.mean((2,3)))
    model.forward = forward_no_opnorm

    model.to(device)

    # 3) Load weights
    checkpoint = torch.load('final_chexnet_finetuned.pth', map_location=device)
    model.load_state_dict(checkpoint)

    # 4) Eval mode
    model.eval()
    return model
model = load_my_model()


########################################################
########################################################
########################################################
########################################################
# STREAMLIT APPLICATION
placeholder.empty()
########################################################
########################################################
########################################################
########################################################

# Set the title
st.title("Medical Bot 🩺🧪🧬")

left, middle, right = st.columns([10, 0.5, 10])

with left:
    st.header("💬 Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Step 1: Handle prompt input **first**
    prompt = st.chat_input("Please enter your message")

    if prompt:
        # Add user's new message immediately
        st.session_state.messages.append({'role': 'user', 'content': prompt, 'avatar': '👩'})

        # Get assistant response
        response = rag_chain.invoke({"input": prompt})['answer']
        st.session_state.messages.append({'role': 'assistant', 'content': response, 'avatar': '👨‍⚕️'})

    # Step 2: Now render the FULL chat history (after adding new ones)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message['role'], avatar=message['avatar']):
                st.markdown(message['content'])


with middle:
    st.markdown(
        """<div style="height: 100%; border-left: 1px solid lightgray;"></div>""",
        unsafe_allow_html=True
    )

with right:
    st.header("📷 IMAGE PNEUMONIA DIAGNOSIS")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True) # use_column_width
        
        with st.spinner('Analyzing image...'):
            time.sleep(2)
            prediction, confidence = predict_xray(img, model)


        st.success(f'{prediction} with confidence {round(confidence,2)}%')