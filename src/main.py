import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def find_best_image_id(news_text, csv_path='descriptions.csv'):
    # Перевод текста новости на английский
    translated_text = translate_to_english(news_text)

    # Загрузка описаний из CSV
    df = pd.read_csv(csv_path)
    descriptions = df['description'].tolist()
    all_texts = descriptions + [translated_text]

    # Векторизация и вычисление сходства
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_index = similarity.argmax()

    # Получение videoid (ID изображения)
    best_videoid = df.iloc[best_index]['videoid']
    best_description = df.iloc[best_index]['description']

    print(f"\nНаиболее подходящее описание (англ.): {best_description}")
    print(f"ID изображения (videoid): {best_videoid}")

if name == "__main__":
    news = input("Введите текст новости на русском: ")
    find_best_image_id(news)
