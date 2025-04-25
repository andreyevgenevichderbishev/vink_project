import argparse
import pandas as pd
from module.processor import Processor

def train_model(path: str, column: str):
    try:
        df = pd.read_csv(path)
        if column not in df.columns:
            print(f"Ошибка: колонка '{column}' не найдена в файле.")
            return
        pr = Processor()
        pr.fit(df[column])
        print(f"Модель успешно обучена по колонке '{column}' из файла '{path}'.")
    except FileNotFoundError:
        print(f"Ошибка: файл '{path}' не найден.")
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")

def predict(product_name: str, top_n: int):
    pr = Processor()
    print(f"Товар конкурента: {product_name}")
    print("Наиболее близкие товары:")
    try:
        for item in pr.predict(product_name, top_n):
            print(item)
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Скрипт для обучения модели и поиска похожих товаров.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('command_or_product', nargs='?', help="Команда 'fit' или название товара конкурента")
    parser.add_argument('topn', nargs='?', type=int, default=5, help='Количество ближайших товаров (по умолчанию 5)')
    parser.add_argument('-p', '--path', help='Путь к CSV-файлу (только для fit)')
    parser.add_argument('-c', '--column', help='Имя колонки с названиями товаров (только для fit)')

    args = parser.parse_args()

    if args.command_or_product == 'fit':
        if not args.path or not args.column:
            print("Ошибка: для обучения укажите и --path, и --column.")
            parser.print_help()
            return
        train_model(args.path, args.column)
    elif args.command_or_product:
        predict(args.command_or_product, args.topn)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()




