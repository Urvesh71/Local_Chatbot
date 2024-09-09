import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import time

# Define models as constants for easy configuration
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "mistral:latest"
BASE_URL = os.getenv("BASE_URL", "http://host.docker.internal:11434")

class PDFChatbot:
    def __init__(self):
        self.setup_directories()
        self.setup_session_state()
        self.display_title()

    def setup_directories(self):
        if not os.path.exists('files'):
            os.mkdir('files')
        if not os.path.exists('vector_database'):
            os.mkdir('vector_database')

    def setup_session_state(self):
        if 'template' not in st.session_state:
            st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

            Context: {context}
            History: {history}

            User: {question}
            Chatbot:"""

        if 'prompt' not in st.session_state:
            st.session_state.prompt = PromptTemplate(
                input_variables=["history", "context", "question"],
                template=st.session_state.template,
            )

        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True,
                input_key="question"
            )

        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = Chroma(persist_directory='vector_database',
                                                  embedding_function=OllamaEmbeddings(base_url=BASE_URL,
                                                                                      model=EMBEDDING_MODEL)
                                                  )

        if 'llm' not in st.session_state:
            st.session_state.llm = Ollama(base_url=BASE_URL,
                                          model=LLM_MODEL,
                                          verbose=True,
                                          callback_manager=CallbackManager(
                                              [StreamingStdOutCallbackHandler()]),
                                          )

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def display_title(self):
        st.title("PDF Chatbot")

    def clear_vectorstore(self):
        if 'vectorstore' in st.session_state:
            del st.session_state['vectorstore']
        if 'retriever' in st.session_state:
            del st.session_state['retriever']
        if 'qa_chain' in st.session_state:
            del st.session_state['qa_chain']
        if 'chat_history' in st.session_state:
            st.session_state.chat_history = []

    def handle_upload(self, uploaded_file):
        if uploaded_file is not None:
            if 'current_pdf' not in st.session_state or st.session_state.current_pdf != uploaded_file.name:
                self.clear_vectorstore()
                st.session_state.current_pdf = uploaded_file.name

                st.info("Uploading and processing your PDF...")

                start_time = time.time()

                bytes_data = uploaded_file.read()
                with open("files/" + uploaded_file.name + ".pdf", "wb") as f:
                    f.write(bytes_data)

                upload_time = time.time() - start_time
                st.success(f"PDF uploaded and saved in {upload_time:.2f} seconds.")

                st.info("Analyzing your document...")

                start_time = time.time()

                loader = PyPDFLoader("files/" + uploaded_file.name + ".pdf")
                data = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len
                )
                all_splits = text_splitter.split_documents(data)

                st.session_state.vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=OllamaEmbeddings(base_url=BASE_URL, model=EMBEDDING_MODEL)
                )
                st.session_state.vectorstore.persist()

                processing_time = time.time() - start_time
                st.success(f"Document analyzed and embeddings created in {processing_time:.2f} seconds.")

                st.session_state.retriever = st.session_state.vectorstore.as_retriever()
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type='stuff',
                    retriever=st.session_state.retriever,
                    verbose=True,
                    chain_type_kwargs={
                        "verbose": True,
                        "prompt": st.session_state.prompt,
                        "memory": st.session_state.memory,
                    }
                )

    def display_chat_history(self):
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])

    def handle_user_input(self):
        user_input = st.chat_input("You:", key="user_input")
        if user_input:
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    start_time = time.time()
                    response = st.session_state.qa_chain(user_input)
                    response_time = time.time() - start_time

                message_placeholder = st.empty()
                full_response = ""
                words = response['result'].split()

                for word in words:
                    full_response += word + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)
                message_placeholder.markdown(full_response)

            st.success(f"Response generated in {response_time:.2f} seconds.")

            chatbot_message = {"role": "assistant", "message": response['result']}
            st.session_state.chat_history.append(chatbot_message)

    def run(self):
        uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
        self.handle_upload(uploaded_file)
        self.display_chat_history()
        self.handle_user_input()

if __name__ == "__main__":
    chatbot = PDFChatbot()
    chatbot.run()
