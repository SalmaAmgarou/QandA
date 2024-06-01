import json
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, DistilBertTokenizer, DistilBertForQuestionAnswering
from typing import List, Tuple

# Define the path to your JSON files
json_files = [
    #"transformed_data_finetune/equipe_recherche.json"
    # "transformed_data_finetune/espace-etudiant-bibliotheque.json",
    # "transformed_data_finetune/espace_entreprise.json",
    # "transformed_data_finetune/espace_etudiant_biblio.json",
    # "transformed_data_finetune/espace_etudiant_clubs.json",
    # "transformed_data_finetune/faculte_conseilEtab.json",
    # "transformed_data_finetune/faculte_contact.json",
    # "transformed_data_finetune/faculte_departements.json",
    # "transformed_data_finetune/faculte_motdoyen.json",
    # "transformed_data_finetune/faculte_presentation.json",
    "transformed_data_finetune/formation_continue.json",
    # "transformed_data_finetune/formation_continue_informations.json",
    # "transformed_data_finetune/formation_initiale.json",
    # "transformed_data_finetune/formation_initiale_information.json",
    # "transformed_data_finetune/fstt.json",
    # "transformed_data_finetune/fstt_actualites.json",
    # "transformed_data_finetune/fstt_service.json"
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

# Initialize the GPT-2 model and tokenizer for question generation
gpt2_model_name = 'gpt2'
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Initialize the DistilBERT model and tokenizer for question answering
qa_model_name = 'distilbert-base-uncased-distilled-squad'
qa_tokenizer = DistilBertTokenizer.from_pretrained(qa_model_name)
qa_model = DistilBertForQuestionAnswering.from_pretrained(qa_model_name)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)


# Function to split content into chunks
def chunk_content(content: str, chunk_size: int = 300) -> List[str]:
    tokens = gpt2_tokenizer.encode(content)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [gpt2_tokenizer.decode(chunk) for chunk in chunks]


# Function to generate questions using GPT-2
def generate_questions(content: str, max_questions: int = 3) -> List[str]:
    prompt = f"Generate {max_questions} questions based on the following content:\n\n{content}\n\nQuestions:"
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')

    # Ensure the length of inputs is within the max length
    if inputs.size(1) > 300:
        inputs = inputs[:, :300]

    outputs = gpt2_model.generate(
        inputs,
        max_new_tokens=50,  # Specify max_new_tokens to control the length of the generated sequence
        num_return_sequences=max_questions,
        no_repeat_ngram_size=2,
        top_p=0.95,
        top_k=50,
        do_sample=True,  # Enable sampling to allow multiple sequences
        pad_token_id=gpt2_tokenizer.eos_token_id  # Set pad token to eos token
    )

    questions = []
    for output in outputs:
        decoded_output = gpt2_tokenizer.decode(output, skip_special_tokens=True)
        if "Questions:" in decoded_output:
            question_part = decoded_output.split("Questions:")[1].strip()
            questions.append(question_part)
        else:
            questions.append(decoded_output.strip())

    return questions


# Generate QA pairs for each content and store them separately
for item in data:
    file_name = item["file_name"]
    content = item["content"]

    # Split content into chunks if it's too long
    content_chunks = chunk_content(content)

    pairs = []
    for chunk in content_chunks:
        # Generate questions using GPT-2 for each chunk
        questions = generate_questions(chunk)
        for question in questions:
            result = qa_pipeline(question=question, context=chunk)
            pairs.append({"Question": question, "Answer": result.get('answer', '')})

    # Create a pandas DataFrame from the generated pairs
    df = pd.DataFrame(pairs)

    # Save the DataFrame to a new JSON file with a name based on the original file
    output_file_name = file_name.split("/")[-1].replace(".json", "_qa.json")
    df.to_json(output_file_name, orient="records")

print("QA pairs generated and saved successfully.")