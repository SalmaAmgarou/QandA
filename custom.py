import json

import pandas as pd
import time
from tqdm import tqdm
import langchain
import re
from typing import List, Tuple

import os

def transform_collection(collection_name, documents):
    transformed_documents = []
    if collection_name == "equipe_recherche":
        for doc in documents:
            membres_info = ""
            membres = doc.get("membres", [])
            for membre in membres:
                membres_info += f"\n- {membre.get('nom', '')} {membre.get('prenom', '')} ({membre.get('email', '')})"

            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "research_team",
                "title": doc["title"],
                "content": "\n".join([
                    '\n'f"laboratoire: {doc.get('laboratoire', '')}",
                    '\n'f"directeur_infos: {doc.get('directeur_infos', '')}",
                    '\n'f"axes_recherche: {' '.join(doc.get('axes_recherche', []))}",
                    '\n'f"projets_recherche_link: {' '.join(doc.get('projets_recherche_link', []))}",
                    '\n'f"these_habil_soutenues_link: {' '.join(doc.get('these_habil_soutenues_link', []))}",
                    '\n'f"prod_scientifique: {' '.join(doc.get('prod_scientifique', []))}",
                    '\n'"membres:" + membres_info,
                    '\n'f"other_membre_key: {doc.get('other_key', '')}"
                ]),
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "espace_entreprise":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "entrepreneurship_space",
                "title": doc["title"],
                "content": "\n".join([
                    '\n'f"qui_sommes_nous: {' '.join(doc.get('qui_sommes_nous', []))}",
                    '\n'f"objectif: {' '.join(doc.get('objectif', []))}",
                    '\n'f"Comment: {' '.join(doc.get('Comment', []))}",
                    '\n'f"activite_service: {' '.join(doc.get('activite_service', []))}"
                ]),
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "fstt_actualites":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "news",
                "title": doc["title"],
                "content": doc["content"],
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "formation_continue":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "continuing_education",
                "title": doc["title"],
                "content": doc["cleaned_content"],
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "formation_initiale":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "initial_education",
                "title": doc["title"],
                "content": doc["Content"],
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "formation_initiale_information":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "initial_education_info",
                "title": doc["title"],
                "content": f"Formation: {doc['Formation']}\nObjectifs: {doc['Objectifs']}\nProgramme: {doc['Programme']}",
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "formation_continue_informations":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "continuing_education_info",
                "title": doc["Formation"],
                "content": f"Filiere: {doc['Filiere']}\nObjectif: {doc['Objectif']}\nPublic_concerne: {doc['Public_concerne']}\nDebouche: {', '.join(doc['debouche'])}",
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "espace_etudiant_biblio":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "student_space_biblio",
                "title": doc["title_biblio"],
                "content": "\n".join(doc["info_biblio"]),
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "espace_etudiant_clubs":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "student_space_club",
                "title": doc["title_club"],
                "content": doc["info_club"],
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "faculte_conseilEtab":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "faculty_council",
                "title": doc["Title"],
                "content": f"Brief: {doc['Brief']}\nResponsabilite: {doc['Responsabilite']}\nName: {doc['Name']}",
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "faculte_contact":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "faculty_contact",
                "title": doc["title"],
                "content": f"Localisation: {doc['localisation']}\nNumero Telephone: {doc['numero_telephone']}\nFax: {doc['fax']}\nEmail: {doc['email']}",
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "faculte_departements":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "faculty_department",
                "title": doc["title"],
                "content": f"Chef: {doc['chef']}\nEmail: {doc['email']}",
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "faculte_motdoyen":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "deans_message",
                "title": doc["title"],
                "content": doc["content"],
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "faculte_presentation":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "faculty_presentation",
                "title": doc["title"],
                "content": doc["content"],
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "fstt_service":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "service",
                "title": doc["service"],
                "content": f"Brief: {doc['Brief']}\nContent: {doc['content']}",
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "fstt_spider":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "announcement",
                "title": doc["title"],
                "content": doc["Content"],
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    elif collection_name == "recherche_struct":
        for doc in documents:
            transformed_doc = {
                "id": str(doc.get("_id", "")),  # Check if "_id" exists in the document
                "type": "research_lab",
                "title": doc["title"],
                "content": doc["laboratoire"],
                "metadata": {
                    "url": doc["url"]
                }
            }
            transformed_documents.append(transformed_doc)
    return transformed_documents

# Define the list of JSON files and their corresponding collection names
json_files = [
    ("data_finetune/espace-etudiant-bibliotheque.json", "espace_etudiant_biblio")
    # ("data_finetune/espace-entreprise.json", "espace_entreprise"),
    # ("data_finetune/equipes_recherche.json", "equipe_recherche"),
    # ("data_finetune/espace-etudiant-clubs.json", "espace_etudiant_clubs"),
    # ("data_finetune/faculte_contact.json", "faculte_contact"),
    # ("data_finetune/faculte_conseil_etablissement.json", "faculte_conseilEtab"),
    # ("data_finetune/faculte_departements.json", "faculte_departements"),
    # ("data_finetune/faculte_presentation.json", "faculte_presentation"),
    # ("data_finetune/formation_continue.json", "formation_continue"),
    # ("data_finetune/formation_continue_informations.json", "formation_continue_informations"),
    # ("data_finetune/formation_initiale.json", "formation_initiale"),
    # ("data_finetune/formation_initiale_informations.json", "formation_initiale_information"),
    # ("data_finetune/fstt.json", "fstt"),
    # ("data_finetune/fstt_actualites.json", "fstt_actualites"),
    # ("data_finetune/fstt_service.json", "fstt_service"),
    # ("data_finetune/mot_doyen.json", "faculte_motdoyen")
]

# Loop through each JSON file and transform the data
# Loop through each JSON file and transform the data
for json_file, collection_name in json_files:
    with open(json_file, "r") as file:
        data = json.load(file)

    transformed_data = transform_collection(collection_name, data)
    # Define the directory path
    directory = "transformed_data_finetune"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the output file name based on collection_name
    output_file = os.path.join(directory, f"{collection_name}.json")
    with open(output_file, "w") as outfile:
        json.dump(transformed_data, outfile, indent=4)  # Replace transformed_data with your actual transformed data
    print(f"Transformed data written to {output_file}")