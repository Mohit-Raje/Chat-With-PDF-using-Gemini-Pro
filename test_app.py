import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


st.set_page_config("Chat With Multiple PDF")
st.header("Chain with pdf")

st.sidebar.title('Enter API Key here')
api_key=st.sidebar.text_input("Enter Google Gemini API Key")


genai.configure(api_key=api_key)
os.environ['GOOGLE_API_KEY']=api_key

##Got PDF
def get_pdf_text(pdf_doc):
    text=""
    for pdf in pdf_doc:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

##Divided into chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000 , chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

#Converted to vectors
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks , embeddings)
    ##Here use Astra DB
    vector_store.save_local("faiss_index")
    
def get_conversational_chain():
    
    promp_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. 
    While providing the answer ensure that you also provide more info regarding the topic on which question is asked 
    but the topic should be from the PDF , provide the information in detailed possible way u can.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro" , temperature=0.3)
    prompt=PromptTemplate(template=promp_template , 
                   input_variables=["context" , "question"])
    chain=load_qa_chain(model , chain_type="stuff" , prompt=prompt)
    return chain


def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index" , embeddings , allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain=get_conversational_chain()
    
    response=chain(
        {"input_documents" : docs , "question":user_question},
        return_only_outputs=True
    )
    
    print(response)
    st.write("Reply: " ,response["output_text"])


user_question=st.text_input("Enter the question")
if user_question:
    user_input(user_question)

with st.sidebar:
    st.title("Menu:")
    pdf_docs=st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    
    if st.button("Submit & Process"):
        with st.spinner("Processing...."):
            raw_text=get_pdf_text(pdf_docs)
            text_chunks=get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")


            
            
    
    
    