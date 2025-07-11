import os
import json
import re
import requests

# Load all 3 JSONs into a single dictionary
DATA_ROOT = "../data"
JSON_FILES = [
    "train_no_dup_with_category_3more_name.json",
    "valid_no_dup_with_category_3more_name.json",
    "test_no_dup_with_category_3more_name.json"
]

combined_json = {}
for filename in JSON_FILES:
    path = os.path.join(DATA_ROOT, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            combined_json.update(data)


def get_item_title(image_path, category):
    """
    Extract the outfit ID and index from image path and return the item name from JSON.
    Example image_path: '../data/images/12345678/4.jpg'
    """
    try:
        match = re.search(r"(\d+)[/\\](\d+)\.jpg", image_path)
        if not match:
            return None

        outfit_id, index_str = match.groups()
        index = int(index_str)

        if outfit_id in combined_json:
            category_data = combined_json[outfit_id].get(category)
            if category_data and category_data.get("index") == index:
                return category_data.get("name")
    except Exception as e:
        print("Error extracting title:", e)

    return None


def get_web_suggestions(query, category):
    """
    Query SerpAPI for shopping suggestions using the item name.
    """
    api_key = "1c3dc38baadaaad92f713b6165aa219ad9713c3d20b4c26adc96697b09c8cf4b"
    params = {
        "engine": "google",
        "q": query,
        "tbm": "shop",
        "api_key": api_key
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        results = response.json()

        shopping_results = results.get("shopping_results", [])
        suggestions = []

        for item in shopping_results[:5]:
            suggestions.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "thumbnail": item.get("thumbnail")
            })

        return suggestions
    except Exception as e:
        print("SerpAPI Error:", e)
        return []
