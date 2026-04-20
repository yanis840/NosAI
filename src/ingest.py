import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()
client = OpenAI()
chroma = chromadb.PersistentClient(path="vectorstore")
collection = chroma.get_or_create_collection("nosai")


def charger_documents(dossier="data/processed"):
    documents = []
    for fichier in os.listdir(dossier):
        if fichier.endswith(".json") and fichier != "boutiques_paris.json":
            chemin = os.path.join(dossier, fichier)
            with open(chemin, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents.append(data)
    print(f"{len(documents)} documents chargés")
    return documents


def creer_chunks(document):
    chunks = []
    maison = document.get("maison", "")
    parfum = document.get("parfum", "")
    base = f"{maison} - {parfum}"

    # Chunk 1 — Identité + notes + description marque
    notes = document.get("notes", {})
    tete = ", ".join(notes.get("tete", []))
    coeur = ", ".join(notes.get("coeur", []))
    fond = ", ".join(notes.get("fond", []))
    prix = document.get("prix", {})
    prix_str = " / ".join([f"{k}: {v}€" for k, v in prix.items() if v])

    chunk1 = f"""Parfum : {base}
Année : {document.get('annee', 'non renseignée')}
Format : {document.get('format', '')}
Famille olfactive : {document.get('famille_olfactive', '')}
Prix : {prix_str}
Notes de tête : {tete}
Notes de cœur : {coeur}
Notes de fond : {fond}
Description marque : {document.get('description_marque', '')}"""

    chunks.append({
        "id": f"{maison}_{parfum}_identite".replace(" ", "_"),
        "texte": chunk1,
        "metadata": {"maison": maison, "parfum": parfum, "type": "identite"}
    })

    # Chunk 2 — Données communautaires
    comm = document.get("communaute", {})
    note = comm.get("note_globale", "")
    votes = comm.get("nb_votes_total", "")

    def pct(d):
        total = sum(d.values())
        if total == 0:
            return ""
        return ", ".join([f"{k}: {round(v/total*100)}%" for k, v in d.items()])

    chunk2 = f"""Parfum : {base}
Note communauté : {note}/5 sur {votes} votes
Appréciation : {pct(comm.get('appreciation', {}))}
Tenue : {pct(comm.get('tenue', {}))}
Sillage : {pct(comm.get('sillage', {}))}
Saison : {pct(comm.get('saison', {}))}
Genre : {pct(comm.get('genre', {}))}
Rapport qualité/prix : {pct(comm.get('rapport_qualite_prix', {}))}
Synthèse : {comm.get('synthese_textuelle', '')}"""

    chunks.append({
        "id": f"{maison}_{parfum}_communaute".replace(" ", "_"),
        "texte": chunk2,
        "metadata": {"maison": maison, "parfum": parfum, "type": "communaute"}
    })

    # Chunk 3 — Évocations
    evocations = comm.get("evocations", [])
    if evocations:
        chunk3 = f"""Parfum : {base}
Évocations de la communauté : {', '.join(evocations)}
Boutiques Paris : {', '.join(document.get('boutiques_paris_propres', []))}
Tags : {', '.join(document.get('tags', []))}"""

        chunks.append({
            "id": f"{maison}_{parfum}_evocations".replace(" ", "_"),
            "texte": chunk3,
            "metadata": {"maison": maison, "parfum": parfum, "type": "evocations"}
        })

    return chunks


def embedder_et_stocker(chunks):
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk["texte"],
            model="text-embedding-3-small"
        )
        vecteur = response.data[0].embedding

        collection.upsert(
            ids=[chunk["id"]],
            embeddings=[vecteur],
            documents=[chunk["texte"]],
            metadatas=[chunk["metadata"]]
        )
        print(f"✓ Stocké : {chunk['id']}")


if __name__ == "__main__":
    print("Chargement des documents...")
    documents = charger_documents()

    print("Création des chunks et embedding...")
    for doc in documents:
        chunks = creer_chunks(doc)
        embedder_et_stocker(chunks)

    print("Terminé — tous les documents sont dans ChromaDB.")
