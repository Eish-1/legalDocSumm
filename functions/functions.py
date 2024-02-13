from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import streamlit as st




template = """
You are a professional documents summarizer, who will 
create a concise and comprehensive summary of the provided 
legal document while adhering to these guidelines:

If you feel that the document uploaded is unethical or not within ethical boundaries
then do not generate 
summary for that.

Craft a summary that is detailed, thorough, in-depth, and complex, 
while maintaining clarity and conciseness.

Incorporate main ideas and essential information, 
eliminating extraneous language and focusing on critical aspects.

Rely strictly on the provided text, without including external information.

Format the summary in paragraphs form for easy understanding.

Remember have a minimum of 2 paragraphs for every Summary.

Remeber to name the category of the document once it's summarize based on your creativity 
at the last in markdown format.
for example:
If the document summary comes like Blog on Nature,
You would say : | Category of the Document | Nature |
If the document summary comes like Home Agreement Paper,
You would say : | Category of the Document | Legal Home Document | 

By following this optimized prompt, you will generate an effective summary
that encapsulates the essence of the given text in a clear, concise, and reader-friendly manner.

{context}
Question:{question}
Summary:

"""

custom_prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
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
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.2)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        verbose=True,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state['chat_history']):
        if i % 2 == 0:
            # with st.chat_message("user"):
                return((message.content))
        else:
            # with st.chat_message("assistant"):
                return(message.content)

