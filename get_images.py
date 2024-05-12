import requests
from bs4 import BeautifulSoup
import os
import sqlite3
from queue import Queue
import threading

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
    """Fetch plant names from the given Wikipedia URL."""
    headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    section_div = soup.find('section', class_='mf-section-0').find('div')

    plants = []
    for ul in section_div.find_all('ul', recursive=False):
        for li in ul.find_all('li', recursive=False):
            text = li.get_text(" ", strip=True)
            plants.append(text)
            print(f"Plant found: {text}")
    return plants

def download_images(task, api_key):
    """Download images for a given plant using Bing Image Search API."""
    plant, train_dir, test_dir = task
    conn, cursor = get_db_connection()

    cursor.execute('SELECT * FROM downloads WHERE plant_name=? AND status="pending"', (plant,))
    pending = cursor.fetchall()

    print(f"Starting download for {plant}. {len(pending)} images pending.")

    for item in pending:
        try:
            img_data = requests.get(item[1])
            img_data.raise_for_status()
            with open(item[3], 'wb') as f:
                f.write(img_data.content)
            cursor.execute('UPDATE downloads SET status="completed" WHERE image_url=?', (item[1],))
            conn.commit()
            print(f"Image downloaded and saved to {item[3]}")
        except Exception as e:
            print(f"Failed to download {item[1]}: {str(e)}")

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

def worker(queue, api_key):
    """Worker function to process download tasks."""
    while not queue.empty():
        task = queue.get()
        download_images(task, api_key)
        queue.task_done()
        print(f"Task completed for {task[0]}")

def check_existing_data():
    """Check if any download tasks are already stored in the database."""
    conn, cursor = get_db_connection()
    cursor.execute('SELECT COUNT(*) FROM downloads')
    count = cursor.fetchone()[0]
    return count > 0

def main():
    """Main function to orchestrate the setup and processing of download tasks."""
    api_key = 'dec361ade8014ad7a4c84262094a76d4'
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
    for _ in range(4):
        t = threading.Thread(target=worker, args=(queue, api_key))
        t.start()

    queue.join()
    print("All tasks completed.")

if __name__ == "__main__":
    main()
