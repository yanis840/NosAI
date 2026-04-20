import os
from dotenv import load_dotenv
from openai import OpenAI
from src.retriever import rechercher

import os
load_dotenv()
if "OPENAI_API_KEY" in __import__("streamlit").secrets:
    os.environ["OPENAI_API_KEY"] = __import__("streamlit").secrets["OPENAI_API_KEY"]
client = OpenAI()

SYSTEM_PROMPT = """Tu es NosAI, un expert en parfumerie de niche.
Tu aides les personnes qui découvrent ou explorent la parfumerie de niche — débutants et curieux bienvenus.

Tes règles absolues :
- Tu distingues toujours la voix de la marque de la voix de la communauté
- Tu cites toujours tes sources et le nombre de votants
- Tu convertis toujours les votes bruts en pourcentages dans tes réponses
- Tu ne fais jamais de recommandation sans données pour la soutenir
- Si tu ne sais pas, tu le dis clairement — jamais d'invention
- Tu corriges poliment si l'utilisateur se trompe
- Tu expliques les termes olfactifs sans jargon inutile


Ton format de réponse :
- Aucun bullet point, aucun tiret, aucun gras — prose uniquement
- Réponse en prose naturelle et fluide
- Structure : voix marque → voix communauté → point de vigilance si nécessaire → boutiques
- Toujours terminer par une question pour continuer la conversation

Ton ton : pédagogique, direct, honnête. Jamais condescendant."""


def repondre(question):
    # Récupérer les chunks pertinents
    resultats = rechercher(question, n_resultats=6)
    chunks = resultats["documents"][0]

    # Construire le contexte
    contexte = "\n\n---\n\n".join(chunks)

    # Appel à GPT-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Voici les données disponibles :

{contexte}

Question de l'utilisateur : {question}

Réponds en te basant uniquement sur ces données."""}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("NosAI — Assistant parfumerie de niche")
    print("=" * 40)
    question = input("Ta question : ")
    print("\nNosAI réfléchit...\n")
    reponse = repondre(question)
    print(reponse)