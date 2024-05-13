import random
import sqlite3
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection():
    """Establishes a database connection and returns the connection and cursor."""
    conn = sqlite3.connect('downloads.db')
    cursor = conn.cursor()
    return conn, cursor

def get_new_image_url(plant, api_key):
    """Fetches a new image URL from Bing Image Search API for a given plant."""
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    params = {
        'q': plant + " plant",
        'count': 1,  # Fetch only one image link
        'offset': random.randint(1, 100),  # Random offset to likely get a different image
        'imageType': 'photo'
    }
    response = requests.get("https://api.bing.microsoft.com/v7.0/images/search", headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['value']:
            return data['value'][0]['contentUrl']
    return None

def update_pending_urls(api_key):
    """Updates the image URLs in the database for entries that are still pending."""
    conn, cursor = get_db_connection()
    cursor.execute("SELECT plant_name, image_url FROM downloads WHERE status='pending'")
    pending_downloads = cursor.fetchall()

    if not pending_downloads:
        logging.info("No pending downloads found.")
        return

    updated_count = 0
    for plant, old_url in pending_downloads:
        new_url = get_new_image_url(plant, api_key)
        if new_url:
            cursor.execute("UPDATE downloads SET image_url=? WHERE image_url=? AND status='pending'", (new_url, old_url))
            conn.commit()
            updated_count += 1
            logging.info(f"Updated URL for {plant}: {old_url} -> {new_url}")
        else:
            logging.warning(f"Failed to fetch new URL for {plant}")

    logging.info(f"Total URLs updated: {updated_count}")
    conn.close()

def main():
    api_key = '3f9cca9ada6743eba8033b9703c8a216'
    update_pending_urls(api_key)

if __name__ == "__main__":
    main()
