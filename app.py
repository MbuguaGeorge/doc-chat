import boto3
from botocore.exceptions import NoCredentialsError
import pickle
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI

# AWS S3 config
s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

def save_vector_store_to_s3(file_name, vector_store):
  bucket_name = os.getenv('bucket-name')
  vector_store_bytes = pickle.dumps(vector_store)
  s3.put_object(Bucket=bucket_name, Key=file_name, Body=vector_store_bytes)

def load_vector_store_from_s3(file_name):
  bucket_name = os.getenv('bucket-name')
  obj = s3.get_object(Bucket=bucket_name, Key=file_name)
  return pickle.loads(obj['Body'].read())

def s3_file_exists(file_name):
  bucket_name = os.getenv('bucket-name')
  try:
      s3.head_object(Bucket=bucket_name, Key=file_name)
      return True
  except:
      return False


# sidebar
with st.sidebar:
  st.title('🗨️ Doc Chat App')
  st.markdown(
    '''
    ## About 
    This application is an LLM-powered chatbot that lets you chat with your PDFs.
    '''
  )
  add_vertical_space(5)
  st.write('Made with ❤️ by [George Mbugua](https://github.com/MbuguaGeorge)')


def main():
  st.header('Chat with PDF 🗨️')

  load_dotenv()

  # upload file
  pdf = st.file_uploader('Upload your PDF', type='pdf')

  if pdf is not None:
    pdf_reader = PdfReader(pdf)
    
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()

    text_splitter= RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
    )

    chunks = text_splitter.split_text(text=text)

    # creating embeddings
    store_name = pdf.name[:-4]

    if s3_file_exists(f"{store_name}.pkl"):
      vectorStore = load_vector_store_from_s3(f"{store_name}.pkl")
    else:
      embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
      vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
      save_vector_store_to_s3(f"{store_name}.pkl", vectorStore)

    query = st.text_input("Ask questions about your file:")

    if query:
      docs = vectorStore.similarity_search(query=query, k=3)

      llm = GoogleGenerativeAI(temperature=0, model="models/text-bison-001")
      chain = load_qa_chain(llm=llm, chain_type='stuff')
      response = chain.run(input_documents=docs, question=query)
      
      st.write(response)


if __name__ == '__main__':
  main()