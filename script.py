import sys
from module.processor import Processor
import pandas as pd

pr = Processor()

arg = sys.argv


if arg[1]=='fit':

    if "," not in arg:
        print('Формат ввода script.py fit PATH , COLUMN')
    else:
        comma_index = arg.index(",")
        path = " ".join(arg[2:comma_index])
        column = " ".join(arg[comma_index+1:])
    
    try:
        df = pd.read_csv(path)
        df = df[column]
        pr.fit(df)
    except FileNotFoundError as e:
        print(f"{e}")
else:        
            
    

    if arg.__len__()==2:
        print("Товар конкурента: "+arg[1])
        print("Наиболее близкие товары: ")
        for i in pr.predict(arg[1]):
            print(i)
    else:
        if arg.__len__()==3:
            print("Товар конкурента: "+arg[1])
            print("Наиболее близкие товары: ")
            for i in pr.predict(arg[1],int(arg[2])):
                print(i) 
        else:
            print('Первый аргумент - название товара')
            print('Второй аргумент - необязательный, число ближайших товаров')
            print('Для обучения модели script fit PATH , COLUMN')
            sys.exit(1)
