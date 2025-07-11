# text_suggestions.py — Generate outfit suggestions using descriptions from dataset (CPU Friendly)

from transformers import pipeline
import json
import os

# Load the FLAN-T5 model (CPU-compatible)
summarizer = pipeline("text2text-generation", model="google/flan-t5-small")

# Load outfit data JSON
DATA_PATH = os.path.join("..", "data", "train_no_dup_with_category_3more_name.json")
with open(DATA_PATH, "r") as f:
    outfit_data = json.load(f)

def get_descriptions_from_ids(outfit_id):
    """
    Given an outfit ID (str), return available item descriptions.
    """
    if outfit_id not in outfit_data:
        return {}
    parts = outfit_data[outfit_id]
    return {cat: parts[cat]['name'] for cat in parts}

def generate_outfit_suggestion_by_id(outfit_id):
    """
    Generate a suggestion using item descriptions from a given outfit ID.
    """
    descriptions = get_descriptions_from_ids(outfit_id)
    if not descriptions:
        return "No descriptions found for the selected outfit ID."

    prompt = ". ".join([f"You are wearing a {desc}" for desc in descriptions.values()])
    prompt += ". Suggest complementary items to complete this look."

    output = summarizer(prompt, max_length=60, do_sample=True)
    return output[0]['generated_text']

# Standalone test
if __name__ == "__main__":
    test_id = "214181831"
    suggestion = generate_outfit_suggestion_by_id(test_id)
    print(f"Outfit ID: {test_id}\nSuggestion: {suggestion}")
