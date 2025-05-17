import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import IsolationForest
import joblib  # برای ذخیره مدل


def get_page_features(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    features = {
        'title': soup.title.string if soup.title else '',
        'text_length': len(soup.get_text()),
        'num_images': len(soup.find_all('img')),
        'num_links': len(soup.find_all('a')),
        'html_length': len(response.text)
    }
    
    return features
