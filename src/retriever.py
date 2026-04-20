import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

import os
load_dotenv()
if "OPENAI_API_KEY" in __import__("streamlit").secrets:
    os.environ["OPENAI_API_KEY"] = __import__("streamlit").secrets["OPENAI_API_KEY"]
client = OpenAI()
chroma = chromadb.PersistentClient(path="vectorstore")
collection = chroma.get_or_create_collection("nosai")


def rechercher(question, n_resultats=3):
    # Transformer la question en vecteur
    response = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    vecteur_question = response.data[0].embedding

    # Chercher les chunks les plus proches dans ChromaDB
    resultats = collection.query(
        query_embeddings=[vecteur_question],
        n_results=n_resultats
    )

    return resultats


if __name__ == "__main__":
    question = "Je cherche un parfum boisé avec du patchouli"
    print(f"Question : {question}\n")

    resultats = rechercher(question)

    for i, doc in enumerate(resultats["documents"][0]):
        print(f"--- Résultat {i+1} ---")
        print(doc[:300])
        print()