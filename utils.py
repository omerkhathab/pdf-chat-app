from dotenv import load_dotenv
import os
import tempfile
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def generate_chunks(files):
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1250, chunk_overlap=250)

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        loader = UnstructuredPDFLoader(tmp_file_path)
        try:
            docs = loader.load()
        except Exception as e:
            print(f"Failed to load PDF {file.name}: {e}")
            continue
        finally:
            os.remove(tmp_file_path)

        for page_num, doc in enumerate(docs, start=1):
            chunks = text_splitter.split_text(doc.page_content)

            for chunk in chunks:
                all_chunks.append({
                    "page_content": chunk,
                    "metadata": {
                        **doc.metadata,
                        "source": file.name,
                        "page": page_num
                    }
                })

    return all_chunks

def create_database(chunks):
    embedding = OllamaEmbeddings(model="nomic-embed-text")

    texts = [chunk["page_content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    vectorDB = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        # persist_directory="./chroma_store",
        collection_name="my_collection"
    )
    return vectorDB

def generate_response(input, files):
    if not input or not files:
        print("Input or Files is missing. Try again")
        return
    
    chunks = generate_chunks(files)
    vectorDB = create_database(chunks)

    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.0
    )

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a helpful AI Assitant. Generate 5 variations of the user's question which will be used
        in querying a vector database for similarity search. Provide these questions separated by newlines
        Original Question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vectorDB.as_retriever(),
        llm=llm,
        prompt=query_prompt
    )

    chat_prompt = ChatPromptTemplate.from_template(
                """Answer the question based on only the given context: {context} Question: {question}"""
            )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(input)
    sources = retriever.invoke(input)
    # removing duplicate chunks:
    # each doc is: Document(
    #     page_content="some chunk of text",
    #     metadata={"source": "file1.pdf", "page": 3}
    # )
    # we'll keep key as page content, value as Document object
    unique_chunks = {}
    for doc in sources:
        unique_chunks[doc.page_content] = doc
    return response, list(unique_chunks.values())