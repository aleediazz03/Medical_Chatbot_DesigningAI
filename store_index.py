from src.helper import load_pdf_file, split_text, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Set up the API key and Pinecone instance
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY


# Extract data from the pdf file
extracted_data = load_pdf_file(data='Data/')
# Split extracted data into chunks
text_chunks = split_text(extracted_data)
# Download the embeddings model
embeddings = download_hugging_face_embeddings()



'''
Cosine similarity is preferred over Euclidean distance or dot product for comparing embeddings because it measures the direction between vectors, not their length. 
This focuses on semantic similarity rather than magnitude, making it more reliable for embeddings where different magnitudes don't necessarily mean different meanings. 
In contrast, Euclidean distance and dot product are affected by vector length, which can distort similarity results.
'''
index_name = "designing-ai-medical-chatbot"
pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", # We can specify the similarity function (pinecone offers others like eucledian or dot-product)
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)



# Embed each text chunk and store the embeddings in our Pinecone database
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks, # This is the data we want to get the embeddings from and store
    index_name=index_name, # The name of the index (database) we will be storing the embeddings
    embedding=embeddings # Our embedding model
)
