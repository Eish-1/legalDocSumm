import os # Added for api key handling
from langchain_groq import ChatGroq # Changed import
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings # Removed OpenAIEmbeddings and unused import
from langchain_huggingface import HuggingFaceEmbeddings # Changed import
from langchain_community.vectorstores import FAISS # Use community FAISS
# from langchain.chat_models import ChatOpenAI # Removed
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import streamlit as st

# --- Constants ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Consistent model

template = """
You are a professional documents summarizer, who will 
create a concise and comprehensive summary of the provided 
legal document while adhering to these guidelines:

If you feel that the document uploaded is unethical or not within ethical boundaries
then do not generate summary for that.

Craft a summary that is detailed, thorough, in-depth, and complex, 
while maintaining clarity and conciseness.

Incorporate main ideas and essential information, 
eliminating extraneous language and focusing on critical aspects.

Rely strictly on the provided text, without including external information.

Format the summary in paragraphs form for easy understanding. Use Markdown for formatting (e.g., paragraphs separated by blank lines).

Remember have a minimum of 2 paragraphs for every Summary.

Remember to name the category of the document once it's summarize based on your creativity 
at the last in markdown format, like this:
| Category of the Document | [Your Creative Category Name] |

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

# --- Functions ---

def get_pdf_text(pdf_path): # Changed input to file path
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n" # Add newline between pages
                else:
                    print(f"Warning: Could not extract text from a page in {pdf_path}")
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        # Optionally re-raise or return None/empty string
    return text

def get_text_chunks(text):
    if not text: # Handle empty input text
        return []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=700, # Reduced chunk size
        chunk_overlap=100, # Adjusted overlap
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Removed get_vectorstore - We can create it within the summarization flow if needed, or main.py can handle it.
# For a summarizer, sometimes just passing all chunks might be feasible if doc is small, or using map_reduce chain.
# Let's simplify generate_summary to use a Map-Reduce approach without vector store first.

# --- New Summarization Function (Map-Reduce approach) ---
from langchain.chains.summarize import load_summarize_chain

def generate_summary(text_chunks, api_key):
    """Generates summary using Map-Reduce chain."""
    if not text_chunks:
        return "Error: No text chunks to summarize."
    if not api_key:
        return "Error: API Key is missing."

    try:
        # Initialize LLM
        llm = ChatGroq(temperature=0.2, groq_api_key=api_key, model_name="llama3-8b-8192") # Example model

        # Load summarization chain (map_reduce type)
        # You might need specific prompts for map and combine steps for better results
        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

        # Convert text chunks back to Document objects for the chain
        from langchain.schema import Document
        docs = [Document(page_content=chunk) for chunk in text_chunks]
        
        # Run the chain using invoke
        # summary = summary_chain.run(docs) # Deprecated
        # The standard input key for load_summarize_chain is 'input_documents'
        # The output is usually a dict with key 'output_text'
        result = summary_chain.invoke({"input_documents": docs})
        summary = result.get('output_text', "Error: Could not extract summary from chain result.")
        return summary

    except Exception as e:
        print(f"Error during summary generation: {e}")
        # Optionally return a more specific error message
        return f"Error generating summary: {e}"

# Removed get_conversation_chain and handle_userinput as they were tied to the RAG approach.
# main.py will now directly call generate_summary.

