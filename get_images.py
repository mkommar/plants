import requests
from bs4 import BeautifulSoup
import os

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


def download_images(plant_names, api_key, num_train=30, num_test=15):
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    endpoint = "https://api.bing.microsoft.com/v7.0/images/search"
    
    base_directory = 'plant_images'
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    
    for plant in plant_names:
        train_dir = os.path.join(base_directory, 'train', plant.replace(' ', '_'))
        test_dir = os.path.join(base_directory, 'test', plant.replace(' ', '_'))
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        total_images_downloaded = 0
        offset = 0
        while total_images_downloaded < (num_train + num_test):
            params = {
                'q': plant + " plant",
                'count': min((num_train + num_test) - total_images_downloaded, 50),
                'offset': offset,
                'imageType': 'photo'
            }
            response = requests.get(endpoint, headers=headers, params=params)
            results = response.json()
            images = results.get('value', [])

            if not images:
                break  # Break if no more images are available
            
            for idx, img in enumerate(images):
                try:
                    img_data = requests.get(img['contentUrl'])
                    img_data.raise_for_status()  # To check for HTTP request errors
                    # Determine if the image is for training or testing
                    if total_images_downloaded < num_train:
                        subdir = train_dir
                    else:
                        subdir = test_dir
                    image_path = os.path.join(subdir, f'{plant.replace(" ", "_")}_{idx + total_images_downloaded}.jpg')
                    with open(image_path, 'wb') as f:
                        f.write(img_data.content)
                except Exception as e:
                    print(f"Failed to download {img['contentUrl']}: {str(e)}")
            
            total_images_downloaded += len(images)
            offset += len(images)  # Increase offset for next batch
            
# Replace 'YOUR_API_KEY' with your Bing API key
api_key = 'dec361ade8014ad7a4c84262094a76d4'
url = "https://simple.m.wikipedia.org/wiki/List_of_plants_by_common_name"
plant_names = get_plant_names(url)
print(plant_names)
download_images(plant_names, api_key)
