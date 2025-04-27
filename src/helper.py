from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings



# Extract all the data from the PDF
def load_pdf_file(data):
    loader = DirectoryLoader(data, 
                             glob='*.pdf', # This makes sure that it only selects pdf files
                             loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents


# Split the extracted data into text chuncks
def split_text(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, # Each chunck will be of 500 characters roughly
                                                   chunk_overlap=20) # Each chunck will overlap 20 characters with the previous one to maintain relevant connections in text
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


'''
The all-MiniLM-L6-v2 model turns sentences into vectors that capture their meaning for tasks like search and matching. 
It’s fast, lightweight, and still very accurate, making it ideal for real-time chatbots. 
We chose it to keep our system quick and efficient without losing quality.
IMPORTANT -> This model returns a 384 dimensional vector as the embedding!
Compared to other models like OpenAI text-embedding-ada-002 (1536 dimensions) or bert-base-nli-mean-tokens (768 dimensions) it is very our model is very lightweight
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
'''
def download_hugging_face_embeddings():
    embeddings=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings