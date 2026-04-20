import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

chroma = chromadb.PersistentClient(path="vectorstore")
collection = chroma.get_or_create_collection("nosai")


def rechercher(question, n_resultats=6):
    import streamlit as st
    from openai import OpenAI
    
    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    response = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    vecteur_question = response.data[0].embedding

    resultats = collection.query(
        query_embeddings=[vecteur_question],
        n_results=n_resultats
    )

    return resultats