from fastapi import FastAPI, Query
import uvicorn
from module.processor import Processor

# Инициализируем процессор
pr = Processor()
app = FastAPI()

# Маршрут для предсказаний с параметром top_n
@app.get('/{name}')
def read_root(
    name: str,
    top_n: int = Query(5, alias="top_n", description="Сколько топ-результатов вернуть"),
):
    """
    Возвращает топ-N похожих товаров для заданного имени.
    name: название товара 
    top_n: сколько похожих возвращать
    """
    # Вызываем метод predict с переданным top_n
    prediction = pr.predict(name, top_n=top_n)
    return {
        "query": name,
        "top_n": top_n,
        "results": prediction
    }

if __name__ == "__main__":
    uvicorn.run("fast_api_app:app", host="0.0.0.0", port=8080, reload=True)

