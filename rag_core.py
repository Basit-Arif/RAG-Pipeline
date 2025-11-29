from __future__ import annotations

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import settings


class RAGPipeline:
    """
    Minimal RAG pipeline:
    - load & split documents
    - build / load Chroma vector store
    - expose a simple .ask(question) method
    """

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please configure it in your environment or .env file.")

        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
        self.llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
        self.vectorstore = None
        self.rag_chain = None

    # ---------- Document Loading & Ingestion ----------

    def _load_documents(self) -> List:
        """
        Load supported documents from the data directory.
        """
        loaders = [
            DirectoryLoader(
                settings.data_dir,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True,
            ),
            DirectoryLoader(
                settings.data_dir,
                glob="**/*.md",
                loader_cls=TextLoader,
                show_progress=True,
            ),
            DirectoryLoader(
                settings.data_dir,
                glob="**/*.csv",
                loader_cls=CSVLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
            ),
            DirectoryLoader(
                settings.data_dir,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
            ),
        ]

        docs: List = []
        for loader in loaders:
            try:
                docs.extend(loader.load())
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Error while loading with loader {loader}: {exc}")
        return docs

    def ingest(self) -> None:
        """
        Load documents from disk, split, embed and persist them into Chroma.
        """
        docs = self._load_documents()
        if not docs:
            print(f"No documents found under '{settings.data_dir}'.")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        split_docs = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=settings.chroma_dir,
        )
        self.vectorstore.persist()
        print(f"Ingestion completed. Stored {len(split_docs)} chunks in '{settings.chroma_dir}'.")

    # ---------- Retrieval + Generation ----------

    def _load_vectorstore(self) -> None:
        """
        Load the existing Chroma vector store from disk.
        """
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_dir,
            )

    def _build_rag_chain(self) -> None:
        """
        Build the LangChain RAG graph (retriever + LLM).
        """
        self._load_vectorstore()

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": settings.k})

        def format_docs(docs) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant that answers questions based **only** on the provided context.
            If the answer is not in the context, say you don't know.

            Context:
            {context}

            Question:
            {question}

            Answer in a clear and concise way.
""".strip()
        )

        rag_inputs = RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )

        self.rag_chain = rag_inputs | prompt | self.llm | StrOutputParser()

    def ask(self, question: str) -> str:
        """
        Ask a question using the RAG pipeline.
        Requires that ingest() has been run at least once to build the vector store.
        """
        if self.rag_chain is None:
            self._build_rag_chain()
        return self.rag_chain.invoke(question)



