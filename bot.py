import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Configure Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return "" #return empty string on error
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key="answer")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        memory=memory,
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="Chat with PDF at Navsahyadri Education ", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with PDF :books: (Gemini)")
    pdf_docs = st.file_uploader("Upload your PDF(s) here and click on 'Process'", accept_multiple_files=True)

    if st.button("Process"):
        with st.spinner("Processing"):
            print(pdf_docs)
            raw_text = get_pdf_text(pdf_docs)
            print(raw_text)
            text_chunks = get_text_chunks(raw_text)

            if not text_chunks:
                st.error("No text found in the uploaded PDFs. Please check your files.")
                return

            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Done!")

    if st.session_state.conversation:
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history = response["chat_history"]

            print("Chat History:", st.session_state.chat_history)
            print("Response Chat History:", response["chat_history"])

            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.write(f"User: {message.content}")
                    else:
                        st.write(f"Bot: {message.content}")
            else:
                st.write("No chat history available.")

if __name__ == '__main__':
    main()
