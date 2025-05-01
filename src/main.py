import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator, MyMemoryTranslator
from sklearn.preprocessing import normalize
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def find_image_by_cosine_similarity(news_text, csv_path):
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

    # Получение videoid 
    best_videoid = df.iloc[best_index]['videoid']
    best_description = df.iloc[best_index]['description']

    print(f"Наиболее подходящее описание: {best_description}")
    print(f"ID изображения (videoid): {best_videoid}")
    print(f"Косинусный коэффициент: {similarity[0][best_index]:.4f}")

def find_image_by_euclidean_distance(news_text, csv_path):
    # Перевод текста новости на английский
    translated_text = translate_to_english(news_text)

    # Загрузка описаний из CSV
    df = pd.read_csv(csv_path)
    descriptions = df['description'].tolist()
    all_texts = descriptions + [translated_text]

    # Векторизация текстов
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Нормализация векторов
    normalized_matrix = normalize(tfidf_matrix)
    
    # Получение вектора для входного текста
    input_vector = normalized_matrix[-1].toarray()
    
    # Вычисление евклидовых расстояний
    distances = []
    for i in range(len(descriptions)):
        desc_vector = normalized_matrix[i].toarray()
        distance = np.linalg.norm(input_vector - desc_vector)
        distances.append(distance)
    
    # Находим индекс с минимальным расстоянием
    best_index = np.argmin(distances)
    
    # Получение videoid
    best_videoid = df.iloc[best_index]['videoid']
    best_description = df.iloc[best_index]['description']
    
    print(f"Наиболее подходящее описание: {best_description}")
    print(f"ID изображения (videoid): {best_videoid}")
    print(f"Евклидово расстояние: {distances[best_index]:.4f}")

def find_image_by_jaccard_similarity(news_text, csv_path):
    # Перевод текста новости на английский
    translated_text = translate_to_english(news_text)
    
    # Загрузка описаний из CSV
    df = pd.read_csv(csv_path)
    descriptions = df['description'].tolist()
    
    # Предварительная обработка текста
    def preprocess_text(text):
        # Приведение к нижнему регистру и удаление знаков препинания
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Токенизация текста
        tokens = word_tokenize(text)
        # Удаление стоп-слов
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w not in stop_words]
        return set(filtered_tokens)
    
    # Получение множества токенов для входного текста
    input_set = preprocess_text(translated_text)
    
    # Вычисление коэффициента Жаккара для каждого описания
    jaccard_scores = []
    for description in descriptions:
        desc_set = preprocess_text(description)
        # Коэффициент Жаккара = размер пересечения / размер объединения
        intersection = len(input_set.intersection(desc_set))
        union = len(input_set.union(desc_set))
        # Избегаем деления на ноль
        jaccard = intersection / union if union > 0 else 0
        jaccard_scores.append(jaccard)
    
    # Находим индекс с максимальным коэффициентом Жаккара
    best_index = np.argmax(jaccard_scores)
    
    # Получение videoid 
    best_videoid = df.iloc[best_index]['videoid']
    best_description = df.iloc[best_index]['description']
    
    print(f"Наиболее подходящее описание: {best_description}")
    print(f"ID изображения (videoid): {best_videoid}")
    print(f"Коэффициент Жаккара: {jaccard_scores[best_index]:.4f}")

def find_image_by_semantic_similarity(news_text, csv_path):
    # Перевод текста новости на английский
    translated_text = translate_to_english(news_text)
    
    # Загрузка описаний из CSV
    df = pd.read_csv(csv_path)
    descriptions = df['description'].tolist()
    
    # Предварительная обработка текста
    def preprocess_text(text):
        # Приведение к нижнему регистру и удаление знаков препинания
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Токенизация текста
        tokens = word_tokenize(text)
        # Удаление стоп-слов
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w not in stop_words]
        return filtered_tokens
    
    # Подготовка корпуса для обучения Word2Vec
    corpus = [preprocess_text(desc) for desc in descriptions]
    corpus.append(preprocess_text(translated_text))
    
    # Обучение модели Word2Vec
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
    
    # Функция для получения вектора документа
    def get_doc_vector(tokens):
        vectors = []
        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])
        if not vectors:
            return np.zeros(100)
        return np.mean(vectors, axis=0)
    
    # Получение векторов для всех документов
    doc_vectors = [get_doc_vector(tokens) for tokens in corpus]
    
    # Вычисление косинусного сходства
    similarities = []
    input_vector = doc_vectors[-1]
    for vector in doc_vectors[:-1]:
        similarity = np.dot(input_vector, vector) / (np.linalg.norm(input_vector) * np.linalg.norm(vector))
        similarities.append(similarity)
    
    # Находим индекс с максимальным сходством
    best_index = np.argmax(similarities)
    
    # Получение videoid (ID изображения)
    best_videoid = df.iloc[best_index]['videoid']
    best_description = df.iloc[best_index]['description']
    
    print(f"Наиболее подходящее описание: {best_description}")
    print(f"ID изображения (videoid): {best_videoid}")
    print(f"Семантическое сходство: {similarities[best_index]:.4f}")

if __name__ == "__main__":
    news = input("Введите текст новости: ")
    csv_path = "dataset/descriptions.csv"  # Путь к файлу с описаниями
    
    print("Выберите метод поиска:")
    print("1. Косинусное сходство")
    print("2. Евклидово расстояние")
    print("3. Коэффициент Жаккара")
    print("4. Семантическое сходство")
    choice = input("Введите номер метода (1, 2, 3 или 4): ")


    
    if choice == "1":
        find_image_by_cosine_similarity(news, csv_path)
    elif choice == "2":
        find_image_by_euclidean_distance(news, csv_path)
    elif choice == "3":
        find_image_by_jaccard_similarity(news, csv_path)
    elif choice == "4":
        find_image_by_semantic_similarity(news, csv_path)
    else:
        print("Неверный выбор. Метод косинусного сходства по умолчанию.")
        find_image_by_cosine_similarity(news, csv_path)
