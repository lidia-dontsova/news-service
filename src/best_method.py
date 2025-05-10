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
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    cosine_score = similarity[0][best_index]

    return {
        'method': 'Косинусное сходство',
        'videoid': best_videoid,
        'description': best_description,
        'score': cosine_score,
        'normalized_score': cosine_score  # Косинусное сходство уже нормализовано от 0 до 1
    }

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
    distance = distances[best_index]
    
    # Нормализация расстояния (преобразуем в сходство от 0 до 1)
    # Используем экспоненциальное преобразование для нормализации
    normalized_score = np.exp(-distance)
    
    return {
        'method': 'Евклидово расстояние',
        'videoid': best_videoid,
        'description': best_description,
        'score': distance,
        'normalized_score': normalized_score
    }

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
    jaccard_score = jaccard_scores[best_index]
    
    return {
        'method': 'Коэффициент Жаккара',
        'videoid': best_videoid,
        'description': best_description,
        'score': jaccard_score,
        'normalized_score': jaccard_score  # Коэффициент Жаккара уже нормализован от 0 до 1
    }

def find_image_by_semantic_similarity(news_text, csv_path):
    # Перевод текста новости на английский
    translated_text = translate_to_english(news_text)
    
    # Загрузка описаний из CSV
    df = pd.read_csv(csv_path)
    descriptions = df['description'].tolist()
    
    # Загрузка предобученной модели
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Получение эмбеддингов для всех текстов
    all_texts = descriptions + [translated_text]
    embeddings = model.encode(all_texts, convert_to_tensor=True)
    
    # Вычисление косинусного сходства
    input_embedding = embeddings[-1]
    similarities = []
    
    for embedding in embeddings[:-1]:
        # Вычисляем косинусное сходство
        similarity = util.pytorch_cos_sim(input_embedding, embedding).item()
        similarities.append(similarity)
    
    # Находим индекс с максимальным сходством
    best_index = np.argmax(similarities)
    
    # Получение videoid (ID изображения)
    best_videoid = df.iloc[best_index]['videoid']
    best_description = df.iloc[best_index]['description']
    semantic_score = similarities[best_index]
    
    return {
        'method': 'Семантическое сходство',
        'videoid': best_videoid,
        'description': best_description,
        'score': semantic_score,
        'normalized_score': semantic_score  # Семантическое сходство уже нормализовано от 0 до 1
    }

def find_best_method(news_text, csv_path="dataset/descriptions.csv"):
    # Получаем результаты всех методов
    results = [
        find_image_by_cosine_similarity(news_text, csv_path),
        find_image_by_euclidean_distance(news_text, csv_path),
        find_image_by_jaccard_similarity(news_text, csv_path),
        find_image_by_semantic_similarity(news_text, csv_path)
    ]
    
    # Выводим результаты всех методов
    print("\nРезультаты всех методов:")
    print("-" * 50)
    for result in results:
        print(f"\nМетод: {result['method']}")
        print(f"Описание: {result['description']}")
        print(f"ID изображения: {result['videoid']}")
        print(f"Обычная оценка: {result['score']:.4f}")
        print(f"Нормализованная оценка: {result['normalized_score']:.4f}")
    
    # Находим лучший метод по нормализованной оценке
    best_result = max(results, key=lambda x: x['normalized_score'])
    
    print("\n" + "=" * 50)
    print(f"ЛУЧШИЙ МЕТОД: {best_result['method']}")
    print(f"Описание: {best_result['description']}")
    print(f"ID изображения: {best_result['videoid']}")
    print(f"Обычная оценка: {best_result['score']:.4f}")
    print(f"Нормализованная оценка: {best_result['normalized_score']:.4f}")
    print("=" * 50)

def analyze_entire_dataset():
    """Анализирует весь датасет BBC News и определяет лучший метод для каждой новости"""
    # Загрузка датасета 
    bbc_df = pd.read_csv("dataset/bbc-news-data.csv", 
                        sep='\t',  
                        encoding='utf-8')
    
    # Берем только первые 10 новостей
    bbc_df = bbc_df.head(10)
    
    # Счетчики для каждого метода
    method_counts = {
        'Косинусное сходство': 0,
        'Евклидово расстояние': 0,
        'Коэффициент Жаккара': 0,
        'Семантическое сходство': 0
    }
    
    # Анализ каждой новости
    print(f"\nАнализ {len(bbc_df)} новостей из датасета BBC...")
    for _, row in tqdm(bbc_df.iterrows(), total=len(bbc_df)):
        news_text = row['content']  
        
        # Получаем результаты всех методов
        results = [
            find_image_by_cosine_similarity(news_text, "dataset/descriptions.csv"),
            find_image_by_euclidean_distance(news_text, "dataset/descriptions.csv"),
            find_image_by_jaccard_similarity(news_text, "dataset/descriptions.csv"),
            find_image_by_semantic_similarity(news_text, "dataset/descriptions.csv")
        ]
        
        # Находим лучший метод
        best_result = max(results, key=lambda x: x['normalized_score'])
        method_counts[best_result['method']] += 1
    
    # Визуализация результатов
    plt.figure(figsize=(12, 6))
    methods = list(method_counts.keys())
    counts = list(method_counts.values())
    
    # Создаем столбчатую диаграмму
    bars = plt.bar(methods, counts)
    
    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    plt.title('Распределение лучших методов по всему датасету BBC News')
    plt.xlabel('Метод')
    plt.ylabel('Количество раз, когда метод был лучшим')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig('best_methods_distribution.png')
    print("\nГрафик сохранен в файл 'best_methods_distribution.png'")
    
    # Выводим статистику
    print("\nСтатистика по методам:")
    total_news = sum(method_counts.values())
    for method, count in method_counts.items():
        percentage = (count / total_news) * 100
        print(f"{method}: {count} раз ({percentage:.1f}%)")

if __name__ == "__main__":
    print("Выберите:")
    print("1. Ввести текст новости вручную")
    print("2. Проанализировать весь датасет BBC News")
    choice = input("Введите номер (1 или 2): ")
    
    if choice == "1":
        news = input("Введите текст новости: ")
        find_best_method(news)
    elif choice == "2":
        analyze_entire_dataset()
    else:
        print("Неверный выбор. Анализ всего датасета.")
        analyze_entire_dataset() 