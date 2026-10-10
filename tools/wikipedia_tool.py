"""
tools/wikipedia_tool.py

Uses Wikipedia API and Wikimedia Commons — completely free, no key needed.
Fetches destination descriptions and real photos.
"""
import requests
import json
from langchain.tools import tool


@tool
def get_destination_info(destination: str) -> str:
    """
    Get detailed information about a travel destination from Wikipedia.
    Input: destination name e.g. "Spiti Valley"
    Returns description, key facts, and travel info.
    """
    try:
        # Search Wikipedia
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action":   "query",
            "list":     "search",
            "srsearch": f"{destination} travel tourism India",
            "format":   "json",
            "srlimit":  1,
        }
        search_resp = requests.get(search_url, params=search_params, timeout=10)
        search_data = search_resp.json()
        results = search_data.get("query", {}).get("search", [])

        if not results:
            return json.dumps({"error": f"No Wikipedia article found for {destination}"})

        page_title = results[0]["title"]

        # Get page extract
        extract_params = {
            "action":   "query",
            "titles":   page_title,
            "prop":     "extracts|pageimages",
            "exintro":  True,
            "explaintext": True,
            "exsentences": 8,
            "piprop":   "thumbnail",
            "pithumbsize": 500,
            "format":   "json",
        }
        extract_resp = requests.get(search_url, params=extract_params, timeout=10)
        extract_data = extract_resp.json()
        pages = extract_data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))

        description = page.get("extract", "No description available.")
        thumbnail   = page.get("thumbnail", {}).get("source", "")

        # Get Wikimedia Commons images
        images_params = {
            "action":   "query",
            "titles":   page_title,
            "prop":     "images",
            "imlimit":  10,
            "format":   "json",
        }
        images_resp = requests.get(search_url, params=images_params, timeout=10)
        images_data = images_resp.json()
        raw_images  = images_data.get("query", {}).get("pages", {})
        raw_images  = next(iter(raw_images.values())).get("images", [])

        # Get actual image URLs from Commons
        photo_urls = []
        if thumbnail:
            photo_urls.append(thumbnail)

        for img in raw_images[:5]:
            img_title = img.get("title", "")
            if any(ext in img_title.lower() for ext in [".jpg", ".jpeg", ".png"]):
                if "flag" not in img_title.lower() and "icon" not in img_title.lower():
                    commons_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{img_title.replace('File:', '').replace(' ', '_')}"
                    photo_urls.append(commons_url)

        return json.dumps({
            "destination": destination,
            "wikipedia_title": page_title,
            "description": description,
            "photos": photo_urls[:4],
            "source": "Wikipedia / Wikimedia Commons",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
