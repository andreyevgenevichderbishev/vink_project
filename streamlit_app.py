import streamlit as st
from module.processor import Processor
import pandas as pd

pr = Processor()

st.header('Vink')
st.subheader('Поиск похожих товаров')

input_name = st.text_input('Введите название товара')
top_n = st.number_input('Введите нужное количество похожих товаров', min_value=1, value=5)

if input_name:
    prediction = pr.predict(input_name,top_n)
    df = pd.DataFrame(prediction, columns=['name', 'score'])
    st.dataframe(df)
