from datasets import load_dataset
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from typing import List, Optional
import os

class VectorStoreManager:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            raise Exception(f"Failed to initialize embeddings: {e}")

    def create_documents(self, texts: List[str]) -> List[Document]:
        docs = []
        for i, text in enumerate(texts):
            try:
                page_content = text
                metadata = {"image_index": i}
                docs.append(Document(page_content=page_content, metadata=metadata))
            except Exception as e:
                print(f"Skipping at index {i}: {e}")
                continue
        print(f"Created {len(docs)} documents from texts")
        print(docs[0])

        return docs

    def build_vector_store(self, texts: List[str], save_path: str) -> Optional[FAISS]:
        try:
            documents = self.create_documents(texts)
            if not documents:
                raise ValueError("No documents created")
            vs = FAISS.from_documents(documents, self.embeddings)
            os.makedirs(save_path, exist_ok=True)
            vs.save_local(save_path)
            print(f"Vector store built and saved to {save_path}")
            return vs
        except Exception as e:
            print(f"Error building vector store: {e}")
            return None

    @st.cache_data
    def load_vector_store(_self, load_path: str) -> Optional[FAISS]:
        try:
            vs = FAISS.load_local(
                load_path,
                _self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Vector_database loaded from {load_path}")
            return vs
        except Exception as e:
            raise ValueError(f"Error loading Vector_database: {e}")


if __name__ == "__main__":
    # 1. Load the dataset
    dataset = load_dataset("tomytjandra/h-and-m-fashion-caption", split="train")

    # 2. Extract the text column into a list
    texts = dataset["text"]

    # 3. Build & save the vector store
    manager = VectorStoreManager()
    vs = manager.build_vector_store(texts, save_path="vector_store")

    # 4. (Optional) reload it to verify
    if vs is not None:
        reloaded = manager.load_vector_store("vector_store")
