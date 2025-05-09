#### Import the libraries
import pandas as pd
import streamlit as st
import joblib

st.title('Expresso Customers Churn Prediction')
st.write('This model use LogisticRegression to make prediction')

# download model
model = joblib.load("expressomodel.pkl")
features = joblib.load("features.pkl")

# Create user input (widgets)
st.header("Input features for prediction")

# The features
region = st.selectbox('Region', ['FATICK', 'DAKAR', 'LOUGA', 'TAMBACOUNDA', 'KAOLACK', 'THIES',
       'SAINT-LOUIS', 'KAFFRINE', 'DIOURBEL', 'MATAM', 'KOLDA',
       'ZIGUINCHOR', 'SEDHIOU', 'KEDOUGOU'])
tenure = st.selectbox('Tenure', ['K > 24 month', 'I 18-21 month', 'H 15-18 month', 'J 21-24 month',
       'F 9-12 month', 'G 12-15 month', 'E 6-9 month', 'D 3-6 month'])
freq_rech = st.number_input('Frequence Reach', min_value = 1, max_value = 150, value = 15)
revenue = st.number_input('Enter revenue', min_value = 1, max_value = 550000, value = 20000)
freq = st.slider('Frequence', 1, 100)
on_net = st.number_input('ON_NET', min_value = 10, max_value = 60000, value = 20)
regular = st.number_input('Regularity', min_value = 1, max_value = 70, value = 12)
freq_top_pack = st.number_input('Freq Top Pack', min_value = 1, max_value = 750, value = 15)
MRG = st.selectbox('MRG', ['NO'])

# Prepare a dataframe
user_input = pd.DataFrame({
    'FREQUENCE_RECH': [freq_rech],
    'REGION': [region],
    'REVENUE': [revenue],
    'FREQUENCE': [freq],
    'ON_NET': [on_net],
    'REGULARITY': [regular],
    'FREQ_TOP_PACK': [freq_top_pack],
    'TENURE': [tenure],
    'MRG': [MRG]
})

# Display input data
st.subheader('Input')
st.write(user_input)

# Encode user_input to match training data
user_input_encoded = pd.get_dummies(user_input, columns = user_input.select_dtypes(include = 'object').columns)
user_input_encoded = user_input_encoded.reindex(columns= features, fill_value=0)

# Predict user_input
if st.button('Predict'):
    prediction = model.predict(user_input_encoded)
    pred_prob = model.predict_proba(user_input_encoded)

    if prediction[0] == 1:
        st.success(f'The probability that the customer will leave is {pred_prob[0][1] * 100:.2f}%')
    else:
        st.success(f'The probability that the customer will stay {pred_prob[0][0] * 100:.2f}%')
