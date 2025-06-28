# PDF Chat App

A simple and interactive web app that lets you upload one or more PDF files and ask questions about their content.

## How It Works (RAG Architecture)

This app follows the **RAG (Retrieval-Augmented Generation)** workflow, where an LLM is *augmented* with relevant information retrieved from your uploaded PDFs. Here's how the process works:

1. **PDF Upload & Chunking** 

   * PDFs are parsed using `pypdf`
   * Text is split into overlapping chunks using `RecursiveCharacterTextSplitter` for better semantic retrieval

2. **Embedding & Storage** 

   * Each chunk is embedded using `OllamaEmbeddings` (`nomic-embed-text` model)
   * Embeddings and metadata are stored in a local vector database using `Chroma`

3. **Query Rewriting & Retrieval** 

   * The user's question is rewritten into **5 semantically different variations** using `llama3-70b-8192`
   * These are used to perform a **multi-query similarity search** via `MultiQueryRetriever`
   * This improves recall by retrieving a richer set of relevant chunks

4. **Response Generation** 

   * Retrieved chunks are passed as **context** to the LLM
   * A final answer is generated using `llama3-70b-8192`
   * The app also displays the **source chunks** and **metadata** so users can verify the answer

---

## Installation
 
Ensure that **[Ollama](https://ollama.com/)** is set up in your system and you have a **[Groq API Key](https://console.groq.com/)**.

1. **Clone the repository**

   ```bash
   git clone https://github.com/omerkhathab/pdf-chat-app.git
   cd pdf-chat-app
   ```

2. **Set up environment variables**

   Create a `.env` file in the root folder:

   ```env
   GROQ_API_KEY=your_key_here
   ````

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```
