import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import IsolationForest
import joblib  # برای ذخیره مدل
import time
import hashlib

def string_to_hash(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 10**8  # یه عدد ۸ رقمی

def get_page_features(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    title = soup.title.string if soup.title else ''
    
    features = {
        'title_hash': string_to_hash(title),
        'text_length': len(soup.get_text()),
        'num_images': len(soup.find_all('img')),
        'num_links': len(soup.find_all('a')),
        'html_length': len(response.text)
    }

    return features

# جمع‌آوری دیتا برای آموزش
#X_train = [get_page_features('http://192.168.1.42/').values() for _ in range(10)]  # چندبار اجرا کن تا نوسانات طبیعی رو بگیری
#X_train = [list(get_page_features('http://192.168.1.42/').values()) for _ in range(10)]
X_train = [list(get_page_features('http://192.168.1.42/').values()) for _ in range(10)]
model = IsolationForest(contamination=0.1)
model.fit(X_train)


# آموزش مدل
#model = IsolationForest(contamination=0.1)
#model.fit(X_train)

# ذخیره مدل
joblib.dump(model, 'deface_detector.pkl')



def check_deface():
    model = joblib.load('deface_detector.pkl')
    current_features = list(get_page_features('http://192.168.1.42/').values())
    prediction = model.predict([current_features])  # 1 = سالم، -1 = ناهنجاری
    
    if prediction[0] == -1:
        print("❗ صفحه احتمالا دیفیس شده! سریع بررسی کن")
    else:
        print("✅ صفحه سالمه")

# هر ۵ دقیقه چک کن
while True:
    check_deface()
    #time.sleep(300)
    print(1)