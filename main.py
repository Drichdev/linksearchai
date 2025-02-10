# python 3.10

import requests
import warnings
from bs4 import BeautifulSoup
from transformers import pipeline, T5Tokenizer
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

def get_page_content(url):
    """
    Récupère le contenu HTML de l'URL et retourne le texte de la page.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Erreur lors du téléchargement de la page : {e}")
        return None

def extract_sections(soup):
    """
    Extrait les rubriques de la page en se basant sur les balises d'en-tête (h1 à h6)
    et en ajoutant les paragraphes qui suivent.
    Si aucun en-tête n'est trouvé, le contenu est rangé sous "Introduction".
    """
    sections = {}
    current_section = None
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            current_section = element.get_text(strip=True)
            sections[current_section] = ""
        elif element.name == 'p':
            texte = element.get_text(strip=True)
            if current_section:
                sections[current_section] += " " + texte
            else:
                sections.setdefault("Introduction", "")
                sections["Introduction"] += " " + texte
    return sections

def main():
    # 1. Récupération de la page web
    url = input("Entrez l'URL de la page (non image) : ").strip()
    html = get_page_content(url)
    if not html:
        return
    soup = BeautifulSoup(html, 'html.parser')
    sections = extract_sections(soup)
    if not sections:
        print("Aucune rubrique détectée. Utilisation du texte complet de la page.")
        full_text = soup.get_text(separator="\n", strip=True)
        sections = {"Texte complet": full_text}

    # Affichage des rubriques détectées
    print("\nRubriques détectées :")
    for i, title in enumerate(sections.keys(), start=1):
        print(f"{i}. {title}")

    choix = input("\nTapez le numéro de la rubrique sur laquelle vous souhaitez poser des questions (ou appuyez sur Entrée pour toutes) : ").strip()
    if choix:
        try:
            idx = int(choix) - 1
            titre_selectionne = list(sections.keys())[idx]
            sections = {titre_selectionne: sections[titre_selectionne]}
            print(f"\nVous avez sélectionné la rubrique : {titre_selectionne}")
        except Exception as e:
            print("Choix invalide, utilisation de toutes les rubriques.")

    # Instanciation explicite du tokenizer avec legacy=False
    qg_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-e2e-qg", legacy=False)

    # 2. Initialisation des pipelines
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-small-e2e-qg", tokenizer=qg_tokenizer)

    # Pour chaque rubrique sélectionnée…
    for section_title, section_text in sections.items():
        print(f"\n=== Rubrique : {section_title} ===")
        if not section_text.strip():
            print("Cette rubrique est vide. Aucun contenu à interroger.")
            continue

        # 3. Génération automatique de questions avec le modèle QG
        print("\nListe des questions/réponses extraites automatiquement :")
        # Pour éviter d'envoyer un texte trop long au modèle QG, on limite le contexte
        max_chars = 1000
        context_for_qg = section_text if len(section_text) <= max_chars else section_text[:max_chars]
        qg_prompt = "generate questions: " + context_for_qg
        try:
            # Génération de plusieurs questions (ici 5, en activant l'échantillonnage)
            generated = qg_pipeline(qg_prompt, max_length=64, num_return_sequences=5, do_sample=True)
        except Exception as e:
            print(f"Erreur lors de la génération des questions : {e}")
            generated = []

        seen_questions = set()  # pour éviter les doublons
        for gen in generated:
            question = gen.get('generated_text', '').strip()
            if not question or question in seen_questions:
                continue
            seen_questions.add(question)
            try:
                qa_result = qa_pipeline(question=question, context=section_text)
                print(f"\nQuestion : {question}")
                print(f"Réponse  : {qa_result['answer']} (Score : {qa_result['score']:.4f})")
            except Exception as e:
                print(f"Erreur lors du traitement de la question '{question}' : {e}")

        # 4. Mode interactif : l'utilisateur peut poser ses propres questions
        print("\nVous pouvez maintenant poser vos questions sur cette rubrique (tapez 'exit' pour passer à la rubrique suivante).")
        while True:
            user_question = input("Question : ").strip()
            if user_question.lower() == 'exit':
                break
            if not user_question:
                continue
            try:
                qa_result = qa_pipeline(question=user_question, context=section_text)
                print(f"Réponse : {qa_result['answer']} (Score : {qa_result['score']:.4f})\n")
            except Exception as e:
                print(f"Erreur lors de la génération de la réponse : {e}")

if __name__ == "__main__":
    main()
