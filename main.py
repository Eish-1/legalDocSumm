import os
from dotenv import load_dotenv
import streamlit as st
from functions import functions as fun # Assuming functions.py is in functions/
from PIL import Image

# --- Constants ---
TEMP_DIR = "temp_data"

# --- Environment & API Key Handling ---
load_dotenv()  # Load variables from .env file

# Prioritize .env, then secrets, then manual input
groq_api_key = os.getenv('GROQ_API_KEY') # Load from .env

if not groq_api_key:
    try:
        # Try loading from Streamlit secrets (if deployed)
        groq_api_key = st.secrets["GROQ_API_KEY"]
        print("Loaded GROQ API key from Streamlit secrets.")
    except KeyError:
        print("GROQ API key not found in .env or Streamlit secrets. Asking user.")
        # Ask user only if not found elsewhere
        groq_api_key = st.text_input(
            'Enter your GROQ_API_KEY: ', 
            type='password',
            key='api_key_input' # Add a key for stability
        )

# Validate API key presence before proceeding
if not groq_api_key:
    st.warning('Please enter your GROQ_API_KEY to use the application.', icon='⚠️')
    st.stop() # Stop execution if no key is available
else:
    # Optionally provide feedback that key is loaded (avoid printing the key itself)
    # st.success("GROQ API Key loaded.")
    # Store the key in session state if needed by other parts, 
    # but preferably pass it directly to functions.
    st.session_state.api_key = groq_api_key

# --- UI Setup ---
st.set_page_config(
    page_title="GenAI Demo | Trigent AXLR8 Labs",
    page_icon='favicon.png',
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Logo
logo_html = """
<style>
    [data-testid="stSidebarNav"] {
        background-image: url(https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png);
        background-repeat: no-repeat;
        background-position: 20px 20px;
        background-size: 80%;
    }
</style>
"""
st.sidebar.markdown(logo_html, unsafe_allow_html=True)
st.title("Legal document summarizer 📗")

st.markdown("""
Summarize your legal documents efficiently.
1. Upload a document (PDF supported).
2. Click 'Summarize'.
""")

# --- File Upload --- 
uploaded_file = st.file_uploader(
    'Upload your document (Currently supports PDF)', type=['pdf'], key='file_uploader'
)

# --- Summarization Trigger --- 
summarize_button = st.button('Summarize Document', key='summarize_button')

# --- Main Logic --- 
if uploaded_file is not None and summarize_button:
    # Ensure temp directory exists
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Save uploaded file temporarily
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info(f"Processing `{uploaded_file.name}`...")

    try:
        # Extract text (assuming this function exists in functions.py)
        doc_text = fun.get_pdf_text(file_path)

        if doc_text:
            # Split text (assuming this function exists)
            text_chunks = fun.get_text_chunks(doc_text)
            
            if text_chunks:
                st.write(f"Document split into {len(text_chunks)} chunks for processing.")

                # Generate summary (assuming this function takes chunks and api key)
                # Pass the confirmed groq_api_key
                summary = fun.generate_summary(text_chunks, groq_api_key)

                st.subheader("Summary:")
                st.success(summary) # Use success box for final output
            else:
                st.error("Could not split document into chunks.")
        else:
            st.error("Could not extract text from the PDF.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        print(f"Error during summarization process: {e}") # Log error
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed temporary file: {file_path}")
            except Exception as e:
                print(f"Error removing temporary file {file_path}: {e}")

elif summarize_button and uploaded_file is None:
    st.warning("Please upload a document first.")


# Remove the unnecessary call to a non-existent main() function
# if __name__=='__main__':
#      main()    

        
