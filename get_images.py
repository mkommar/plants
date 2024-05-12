import requests
from bs4 import BeautifulSoup
import os
import sqlite3
from queue import Queue
import threading
# Define thread-local storage
thread_local = threading.local()

def get_db_connection():
    """Retrieve a new database connection for the current thread."""
    if not hasattr(thread_local, "connection"):
        thread_local.connection = sqlite3.connect('downloads.db', check_same_thread=False)
        thread_local.cursor = thread_local.connection.cursor()
    return thread_local.connection, thread_local.cursor

def setup_database():
    """Create the database and downloads table if not already present."""
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
    plant, train_dir, test_dir = task
    conn, cursor = get_db_connection()  # Get thread-specific DB connection
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    endpoint = "https://api.bing.microsoft.com/v7.0/images/search"
    offset = 0
    num_images = 45  # Total images to download

    while True:
        # Check database for remaining downloads
        cursor.execute('SELECT * FROM downloads WHERE plant_name=? AND status="pending"', (plant,))
        pending = cursor.fetchall()
        if not pending:
            break  # No more pending downloads

        for item in pending:
            try:
                img_data = requests.get(item[1])
                img_data.raise_for_status()
                subdir = train_dir if 'train' in item[3] else test_dir
                with open(item[3], 'wb') as f:
                    f.write(img_data.content)
                cursor.execute('UPDATE downloads SET status="completed" WHERE image_url=?', (item[1],))
                conn.commit()
            except Exception as e:
                print(f"Failed to download {item[1]}: {str(e)}")

def prepare_downloads(plant_names, api_key):
    conn, cursor = get_db_connection()  # Get thread-specific DB connection

    for plant in plant_names:
        train_dir = os.path.join('plant_images', 'train', plant.replace(' ', '_'))
        test_dir = os.path.join('plant_images', 'test', plant.replace(' ', '_'))
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Prepare download tasks and store in the database
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
            cursor.execute('INSERT INTO downloads (plant_name, image_url, status, file_path) VALUES (?, ?, ?, ?)',
                      (plant, img['contentUrl'], 'pending', file_path))
        conn.commit()

def worker(queue, api_key):
    while not queue.empty():
        task = queue.get()
        download_images(task, api_key)
        queue.task_done()

def main():
    api_key = 'dec361ade8014ad7a4c84262094a76d4'
    url = "https://simple.m.wikipedia.org/wiki/List_of_plants_by_common_name"
    setup_database()
    plant_names = get_plant_names(url)
    prepare_downloads(plant_names, api_key)
    
    queue = Queue()
    for plant in plant_names:
        train_dir = os.path.join('plant_images', 'train', plant.replace(' ', '_'))
        test_dir = os.path.join('plant_images', 'test', plant.replace(' ', '_'))
        queue.put((plant, train_dir, test_dir))

    for _ in range(4):  # Number of threads
        t = threading.Thread(target=worker, args=(queue, api_key))
        t.start()

    queue.join()

if __name__ == "__main__":
    main()
