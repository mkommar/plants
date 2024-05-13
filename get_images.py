import random
import requests
from bs4 import BeautifulSoup
import os
import sqlite3
from queue import Queue
import threading
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define thread-local storage for database connections to ensure thread safety
thread_local = threading.local()

def get_db_connection():
    """Retrieve or create a database connection for the current thread."""
    if not hasattr(thread_local, "connection"):
        thread_local.connection = sqlite3.connect('downloads.db', check_same_thread=False)
        thread_local.cursor = thread_local.connection.cursor()
        print("Database connection created for thread:", threading.current_thread().name)
    return thread_local.connection, thread_local.cursor

def setup_database():
    """Create the downloads table in the database if it does not already exist."""
    conn, cursor = get_db_connection()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS downloads (
        plant_name TEXT,
        image_url TEXT,
        status TEXT,
        file_path TEXT
    )
    ''')
    conn.commit()
    print("Database setup completed.")

def get_plant_names(url):
    headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    section_div = soup.find('section', class_='mf-section-0').find('div')

    plants = []

    def extract_names(li):
        # Extract clean text from each <li> and handle nested <ul>
        text = li.get_text(" ", strip=True)  # Using a space as a separator
        nested_ul = li.find('ul')
        if nested_ul:
            # Replace text of nested <ul> with nothing to avoid duplication
            nested_text = nested_ul.get_text(" ", strip=True)
            text = text.replace(nested_text, "").strip()
        return text

    for ul in section_div.find_all('ul', recursive=False):
        for li in ul.find_all('li', recursive=False):
            text = extract_names(li)
            if text:
                plants.append(text)
            # Also process any nested <ul> within the <li>
            nested_ul = li.find('ul')
            if nested_ul:
                for nested_li in nested_ul.find_all('li', recursive=False):
                    nested_text = extract_names(nested_li)
                    if nested_text:
                        plants.append(nested_text)
                    
    return plants

def download_images(task, api_key):
    """Download images for a given plant using Bing Image Search API with retry on 403 errors, handling 406 errors."""
    plant, train_dir, test_dir = task
    conn, cursor = get_db_connection()

    cursor.execute('SELECT * FROM downloads WHERE plant_name=? AND status="pending"', (plant,))
    pending_images = cursor.fetchall()

    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; YourBot/1.0; +http://yourdomain.com/bot.html)',
        'Accept': 'Accept: image/*'  # Accept any media type
    }

    for image in pending_images:
        image_url, file_path = image[1], image[3]
        try:
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()  # Check for HTTP errors
            if not os.path.exists(os.path.dirname(image[3])):
                os.makedirs(os.path.dirname(image[3]))
                logging.info(f"Directory created: {os.path.dirname(image[3])}")

            with open(file_path, 'wb') as f:
                f.write(response.content)
            cursor.execute('UPDATE downloads SET status="completed" WHERE image_url=?', (image_url,))
            conn.commit()
            logging.info(f"Image downloaded and saved to {file_path}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logging.warning(f"403 Forbidden received for URL: {image_url}. Retrying with a new URL.")
                # Additional logic to handle retries or fetch new URLs
            elif e.response.status_code == 406:
                logging.error(f"406 Not Acceptable Error for URL: {image_url}. Check the 'Accept' headers.")
            else:
                logging.error(f"Failed to download {image_url}: {e}")
        except Exception as e:
            logging.error(f"General error during download for {image_url}: {e}")

def get_new_image_url(plant, api_key):
    """Fetch a new image URL from Bing Image Search API."""
    params = {
        'q': f"{plant} plant",
        'count': 1,
        'offset': random.randint(1, 100),  # Random offset to likely get a different image
        'imageType': 'photo',
        'mkt': 'en-US'
    }
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    response = requests.get("https://api.bing.microsoft.com/v7.0/images/search", headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['value']:
            return data['value'][0]['contentUrl']
    return None

def prepare_downloads(plant_names, api_key):
    """Prepare download tasks for each plant and store them in the database."""
    conn, cursor = get_db_connection()
    print("Preparing download tasks.")

    for plant in plant_names:
        train_dir = os.path.join('plant_images', 'train', plant.replace(' ', '_'))
        test_dir = os.path.join('plant_images', 'test', plant.replace(' ', '_'))
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        response = requests.get("https://api.bing.microsoft.com/v7.0/images/search", headers={
            'Ocp-Apim-Subscription-Key': api_key
        }, params={
            'q': plant + " plant",
            'count': 45,
            'imageType': 'photo'
        })
        results = response.json()

        for idx, img in enumerate(results.get('value', [])):
            file_path = os.path.join(train_dir if idx < 30 else test_dir, f"{plant.replace(' ', '_')}_{idx}.jpg")
            cursor.execute('INSERT INTO downloads (plant_name, image_url, status, file_path) VALUES (?, ?, "pending", ?)',
                      (plant, img['contentUrl'], file_path))
        conn.commit()
        print(f"Download tasks prepared for {plant}")

# Example of logging within the worker function
def worker(queue, api_key):
    while not queue.empty():
        task = queue.get()
        logging.debug(f"Processing task: {task}")
        try:
            download_images(task, api_key)
            queue.task_done()
            logging.info(f"Task completed successfully for {task[0]}")
        except Exception as e:
            logging.error(f"Error processing task {task[0]}: {e}")
            queue.task_done()  # Ensure task_done is called even on error


def check_existing_data():
    """Check if any download tasks are already stored in the database."""
    conn, cursor = get_db_connection()
    cursor.execute('SELECT COUNT(*) FROM downloads')
    count = cursor.fetchone()[0]
    return count > 0

def main():
    """Main function to orchestrate the setup and processing of download tasks."""
    api_key = '3f9cca9ada6743eba8033b9703c8a216'
    url = "https://simple.m.wikipedia.org/wiki/List_of_plants_by_common_name"
    setup_database()

    if not check_existing_data():
        plant_names = get_plant_names(url)
        prepare_downloads(plant_names, api_key)
    else:
        print("Existing data found. Skipping data preparation.")

    queue = Queue()
    for plant in get_plant_names(url):
        train_dir = os.path.join('plant_images', 'train', plant.replace(' ', '_'))
        test_dir = os.path.join('plant_images', 'test', plant.replace(' ', '_'))
        queue.put((plant, train_dir, test_dir))

    print("Starting worker threads.")
    for _ in range(256):
        t = threading.Thread(target=worker, args=(queue, api_key))
        t.start()

    queue.join()
    print("All tasks completed.")

if __name__ == "__main__":
    main()
