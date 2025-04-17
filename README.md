# Product Similarity Toolkit

Набор инструментов для обучения модели поиска похожих товаров и разных способов её использования: модуль Python, CLI-скрипт, Streamlit-приложение и FastAPI-сервис.

---

## 📁 Структура проекта

```
project/
├── module/
│   └── processor.py      # Класс Processor с методами fit и predict
├── scripts/
│   └── cli.py            # Скрипт для командной строки (CLI)
├── app_streamlit.py      # Streamlit-приложение
├── fast_api_app.py       # FastAPI-приложение
├── requirements.txt      # Список зависимостей
└── README.md             # Эта документация
```

---

## 🚀 Установка

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/ваш_проект/project.git
   cd project
   ```
2. Создать и активировать виртуальное окружение:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\\Scripts\\activate    # Windows
   ```
3. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

### 📄 Содержимое `requirements.txt`
```text
fastapi>=0.75.0
uvicorn[standard]>=0.18.0
pandas>=1.0.0
scikit-learn>=0.24.0
joblib>=1.0.0
streamlit>=1.0.0
```

---

## ⚙️ Использование

### 1. Модуль `Processor`

Импорт и вызов методов:

```python
from module.processor import Processor
import pandas as pd

pr = Processor()
# Обучение
series = pd.Series(["товар1", "товар2", "тест"])
pr.fit(series)
# Предсказание
results = pr.predict("товар1", top_n=5)
print(results)  # [("товар2", 0.87), ...]
```

### 2. CLI-скрипт

```bash
python scripts/cli.py товар1 товар2 , категория1 категория2
```
- Всё до запятой — названия товаров, после — категории.

### 3. Streamlit-приложение

Запустить интерфейс в браузере:
```bash
streamlit run app_streamlit.py
```
Интерактивно обучить модель и выполнять поиск.

### 4. FastAPI-сервис

```bash
uvicorn fast_api_app:app --reload --host 0.0.0.0 --port 8080
```

- **GET** `/predict?name=<товар>&top_n=<число>`

Пример:
```
GET http://localhost:8080/predict?name=банан&top_n=3
```

---

## 🛠️ Технические детали

- **Ядро**: `module/processor.py` — TF-IDF + NearestNeighbors.
- **Сохранение** моделей в папке `models/`.
- **CLI** использует парсинг аргументов и объединяет до/после запятой.
- **Streamlit** для простого UI.
- **FastAPI** для REST API.
