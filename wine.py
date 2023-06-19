import pickle
import streamlit as st
import numpy as np

# membaca model
model = pickle.load(open('wine.sav', 'rb'))
scaler = pickle.load(open('scaler_wine.sav','rb'))

#judul web
st.title('Prediksi Kualitas Wine')

#membagi kolom
col1, col2 = st.columns(2)

with col1 :
    fixed_acidity = st.number_input('Input  ilai keasaman tetap')
    volatile_acidity = st.number_input('Input nilai keasaman mudah menguap')
    citric_acid = st.number_input('Input nilai asam sitrat')
    residual_sugar = st.number_input('Input nilai sisa gula')
    chlorides = st.number_input('Input nilai klorida')
    free_sulfur_dioxide = st.number_input('Input nilai sulfur dioksida bebas')
    total_sulfur_dioxide = st.number_input('Input nilai total sulfur dioksida')

with col2 :
    density = st.number_input('Input nilai densitas')
    pH = st.number_input('Input nilai pH')
    sulphates = st.number_input('Input nilai sulfat')
    alcohol = st.number_input('Input nilai alkohol')

# code untuk prediksi
prediction = ''
input_data = (fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
              chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
              pH, sulphates, alcohol)

input_data_as_numpy_array = np.array(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)

# membuat tombol untuk prediksi
if st.button('Proses'):
    wine_prediction = model.predict(std_data)
    if(wine_prediction[0] == 0):
        prediction = 'Kualitas Wine Buruk'
    else:
        prediction = 'Kualitas Wine Bagus'
    st.success(prediction)
