from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import re
import os


def extract_tokens(text):
    """
    Функция разбивает подаваемый текст на токены:
      - Числа (целые и дробные, разделитель может быть точкой или запятой),
        при этом:
          * Отбрасываются числа, начинающиеся с 0.
          * Отбрасываются числа, после которых через пробел идёт 'кг'.
      - Слова (русские и латинские, если их длина > 3 символов, приводятся к нижнему регистру).
      - Отдельностоящие заглавные латинские буквы.
    
    Аргумент:
	text (str) текстовая строка.
    Возвращает:
	list[str] список токенов.
    """
    # Регулярное выражение:
    # (?!0\d)                - число не должно начинаться с 0 (например, 0123 не подходит)
    # \d+(?:[.,]\d+)?        - число с необязательной дробной частью
    # (?!\s+кг)              - сразу после числа не должно идти пробельное пространство и "кг"
    # [A-Za-zА-Яа-я]+        - последовательности букв (русские и латинские)
    # \b[A-Z]\b              - отдельностоящие заглавные латинские буквы
    pattern = pattern = r'(?!0\d)\d+(?:[.,]\d+)?(?!\s+кг)|[A-Za-zА-Яа-я]+|(?:\b[A-Z]\b|(?<=\d)[A-Z]|[A-Z](?=\d))'
    
    tokens = re.findall(pattern, text)
    result = []
    
    for token in tokens:
        # Если токен начинается с цифры – это число
        if re.match(r'^\d', token):
            result.append(token)
        # Если токен – это отдельная заглавная латинская буква, оставляем его без изменений
        elif re.match(r'^[A-Z]$', token):
            result.append(token)
        else:
            token_lower = token.lower()
            # Добавляем слово, если его длина больше 3 символов
            if len(token_lower) > 3:
                result.append(token_lower[:4])
    return result

# Кастомный токенайзер для TfidfVectorizer
def custom_tokenizer(text):
    return extract_tokens(text)

class Processor():

    def __init__(self):
        
        self.vectorizer = None
        self.nn_model = None
        self.df = None

    def _load(self):
        """
        Загружает векторизатор, nn_model и df только один раз.
        """
        if self.vectorizer is None or self.nn_model is None or self.df is None:
            # проверяем наличие файлов
            vect_path = 'models/vectorizer.joblib'
            nn_path   = 'models/nn_model.joblib'
            df_path   = 'models/df.csv'
            if not (os.path.exists(vect_path) and os.path.exists(nn_path) and os.path.exists(df_path)):
                raise FileNotFoundError(
                    "Файлы модели не найдены. Сначала вызовите fit(), чтобы их создать."
                )
            # загрузка
            self.vectorizer = joblib.load(vect_path)
            self.nn_model    = joblib.load(nn_path)
            self.df          = pd.read_csv(df_path)    
        
    def fit(self, df : pd.Series ):
        
        """
        Обучает модель на списке названий товаров. Перед этим убираются дубликаты. Обученная модель сохраняется:
	  models/vectorizer.joblib TF-IDF векторизатор
	  models/nn_model.joblib модель ближайщих соседей
	  models/df.csv список названий без дубликатов.
	Аргумент:
	  df (pd.Series) список названий товаров
	Возвращает:
	  ничего
	
        """

        #Убираем дубликаты и тестовые записи

        df = df.to_frame(name='name')
        df = df.drop_duplicates()
        df = df.loc[~df['name'].isin(['тест2', 'тест4', 'v', 'test', 'тест', 'ТЕСТ'])].dropna()

        
        # Построение TF‑IDF векторизатора на наших товарах (df)
        vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
        x = vectorizer.fit_transform(df['name'])
  
        # Создаем модель NearestNeighbors для поиска ближайших товаров,
        # используем косинусное расстояние
        nn_model = NearestNeighbors(metric='cosine')
        nn_model.fit(x)
        # Проверка наличия папки.
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(vectorizer,'models/vectorizer.joblib')
        joblib.dump(nn_model,'models/nn_model.joblib')
        df.to_csv('models/df.csv')
        print('Model fitted')

    def predict(self, text, top_n=5):

        """
        Находит top_n ближайщих товаров к данному.

	Аргумент:
	  text (str) название товара для которого ищем ближайшие соответствия.
	  top_n (int, по умолчанию 5) число ближайших товаров, которые нужно вывести.
	Возвращает:
	  list[(name, similarity)], где name (str) имя товара, similarity(real) степень сходства , similarity = 1 - cosine_distance.  
        """
        
        try:
            self._load()
        except FileNotFoundError as e:
            print(f"[Warning] {e}")
            return []
        except Exception as e:
            print(f"[Error] При загрузке модели: {e}")
            return []
        
        vector = self.vectorizer.transform([text])
        distances, indices = self.nn_model.kneighbors(vector,top_n)
        # Преобразуем расстояния в сходство: чем меньше расстояние, тем больше сходство
        similarities = 1 - distances[0]
        predicted_products = self.df.iloc[indices[0]]["name"].tolist()
        # Формируем список кортежей (название, сходство)
        return list(zip(predicted_products, similarities))
