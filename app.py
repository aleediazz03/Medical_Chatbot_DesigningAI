import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
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
st.title("Medical Bot")

previous = '''# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role'], avatar=message['avatar']):
        st.markdown(message['content'])

# React to user input
prompt = st.chat_input("Please enter your message")
if prompt: 
    # Display user message in chat message container
    with st.chat_message(name='user', avatar='👩'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({'role':'user', 'content':prompt, 'avatar':'👩'})

    response = rag_chain.invoke({"input": prompt})['answer']
    # Display assistant response in chat container
    with st.chat_message(name='assistant', avatar='👨‍⚕️'):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({'role':'assistant', 'content':response, 'avatar':'👨‍⚕️'})


st.divider()
st.header("Upload an Image for Analysis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    with st.spinner('Analyzing image...'):
        time.sleep(2)  # simulate processing
        prediction = "Diagnosis: Healthy skin"  # Replace this with your real model

    st.success(prediction)
'''



left, middle, right = st.columns([10, 0.5, 10])

with left:
    st.header("💬 Chatbot Section")

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
    st.header("📷 Image Upload Section")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner('Analyzing image...'):
            time.sleep(2)
            prediction = "Diagnosis: Healthy skin"

        st.success(prediction)