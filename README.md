# Legal Document Summarizer 📄

## Overview

The Legal Document Summarizer is a Streamlit-based application that leverages OpenAI's API to process legal documents and generate summaries. It allows users to upload PDF files,
processes the content, and provides a summarized version of the document without having Images.

## Getting Started

### Prerequisites

- Python 3.11.5
- Pip package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/legal-docs-summarizer.git


Navigate to the project directory:

cd legalDocSumm

1. install uv and set up our Python project and environment

   ```bash
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
 Make sure to restart your terminal afterwards to ensure that the uv command gets picked up.
 
3. Install all the requirements using: 
   uv sync


4. **Set Your Groq API Key** 🔑

   Open `.env` and add your Groq API key:

   ```bash
   GROQ_API_KEY=your-api-key-here
   ```

5. **Run the Application** 🚀

   ```bash
   uv run streamlit run app.py
   ```

Conclude your notes with [End of Notes, Message #X] 
to indicate completion, where "X" represents the total number of messages
that I have sent. In other words, include a message counter where you
start with #1 and add 1 to the
message counter every time I send a message.
