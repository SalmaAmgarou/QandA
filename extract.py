import json
import pandas as pd
from transformers import pipeline, BertTokenizer, BertLMHeadModel
from typing import List, Tuple
import pandas as pd
import time
from tqdm import tqdm
import langchain
import re

# Initialize the language model
llm = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

import os
# Define the path to your JSON files
json_files = [
    "transformed_data_finetune/equipe_recherche.json",
    "transformed_data_finetune/espace-etudiant-bibliotheque.json",
    "transformed_data_finetune/espace_entreprise.json",
    "transformed_data_finetune/espace_etudiant_biblio.json",
    "transformed_data_finetune/espace_etudiant_clubs.json",
    "transformed_data_finetune/faculte_conseilEtab.json",
    "transformed_data_finetune/faculte_contact.json",
    "transformed_data_finetune/faculte_departements.json",
    "transformed_data_finetune/faculte_motdoyen.json",
    "transformed_data_finetune/faculte_presentation.json",
    "transformed_data_finetune/formation_continue.json",
    "transformed_data_finetune/formation_continue_informations.json",
    "transformed_data_finetune/formation_initiale.json",
    "transformed_data_finetune/formation_initiale_information.json",
    "transformed_data_finetune/fstt.json",
    "transformed_data_finetune/fstt_actualites.json",
    "transformed_data_finetune/fstt_service.json"
]

# Read JSON files and extract content from the "content" field
data = []
for json_file in json_files:
    with open(json_file, "r") as file:
        json_data = json.load(file)
        for item in json_data:
            content = item.get("content", "")
            if content:
                data.append({"file_name": json_file, "content": content})
# Initialize the model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertLMHeadModel.from_pretrained(model_name,is_decoder=True)

for item in data:
    file_name = item["file_name"]
    content = item["content"]

focus = None
describe = None
prompt = f"""### Instruction: Based on the {focus} information of the list of the contents below,  generate 10 instruction-detailed response pairs.
  Make sure the Instruction-Response are in the json format:\n\n
  ### Example: {{"Instruction": "the instruction", "Response": "the response"}}\n\n
  ### Description:{describe}\n\n
  ### Response:"""
def extract_instruction_response_pairs(string: str)-> Tuple[List[str], List[str]]:
    """
    Extracts pairs of instructions and responses from a JSON-formatted string.

    Parameters:
        - json_string (str): A string containing JSON-formatted instruction and response pairs.

    Returns:
        - instructions (list): A list of extracted instructions.
        - responses (list): A list of extracted responses corresponding to the instructions.
    """

    pattern = r'{"Instruction": "(.*?)", "Response": "(.*?)"}'

    # Use re.findall to extract matches
    matches = re.findall(pattern, string)

    # Extract lists of "Instruction" and "Response"
    instructions = [match[0] for match in matches]
    responses = [match[1] for match in matches]

    return instructions, responses

All_instructions = []
All_reponses = []
start = time.time()

for idx in tqdm(range(len(data))):
    content = data[idx]["content"]
    focus = "introductory"  # You can change this based on your content analysis
    describe = content  # Use the content from your JSON file here
    prompt = f"""### Instruction: Based on the {focus} information of the content below, generate 5 instruction-detailed response pairs.
    Make sure the Instruction-Response are in the json format:\n\n
    ### Example: {{"Instruction": "the instruction", "Response": "the response"}}\n\n
    ### Description:{describe}\n\n
    ### Response:"""
    generated_text = llm(prompt)
    ins, res = extract_instruction_response_pairs(generated_text)
    All_instructions.extend(ins)
    All_reponses.extend(res)

print("\n\n===Time: {} seconds===".format(time.time()-start))