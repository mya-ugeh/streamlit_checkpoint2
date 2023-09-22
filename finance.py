import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib as plt
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import joblib
import pickle
import time


#load the model
mod = pickle.load(open('randomForest_model.pkl', 'rb'))

#to add picture from local computer
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('5917.jpg') 

# to import css file into streamlit
with open('finance.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


st.markdown("<h1>PREDICTING FININACIAL INCLUSION IN EAST AFRICA</h1>", unsafe_allow_html=True)

st.title("Project Overview")

st.markdown("<p style = 'color : black'>Financial inclusion is a critical factor in a region's economic development, and East Africa is no exception. This project aims to leverage machine learning to tackle an important problem: predicting which individuals in East Africa are most likely to have or use a bank account. By identifying the key factors that influence financial inclusion, we can better understand the dynamics of financial services access in the region and work towards improving it.</p>", unsafe_allow_html=True)



data = pd.read_csv('Financial_inclusion_dataset.csv')
df = pd.read_csv('Variable.csv')
df.reset_index(drop=True, inplace=True)
df.drop('Unnamed: 0', inplace = True, axis = 1)


custom_css = """
<style>
    .st-af {
        color: white !important; 
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)



with st.form('my_form', clear_on_submit=True):
    st.header("BANK ACCOUNT PREDICTION")
    with st.expander("Variable Definitions"):
        st.table(df)
    phone = st.selectbox('Cellphone Acess',['', 'Yes', 'No'])
    education = st.selectbox('Education Level',['',
                                           'Primary education',
                                           'Secondary education',
                                           'Tertiary education',
                                           'Vocational/Specialised training',
                                           'No formal education',
                                           'Other/Dont know/RTA'])
    job = st.selectbox('Job Type',['','Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income'])
    country = st.selectbox('Country', ['','Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
    gender = st.selectbox('Gender',['','Female','Male'])
    year = st.selectbox('Year', ['',2018, 2016, 2017])
    
    submitted = st.form_submit_button("SUBMIT")   
    if (phone and education and job and country and gender and year):
        if submitted:
            with st.spinner(text='In progress'):
                time.sleep(3)
                st.write("Your Inputted Data:")
                input_var = pd.DataFrame([{'cellphone_access' : phone,'education_level' : education,'job_type' : job,'country' : country, 'year' : year, 'gender_of_respondent' : gender}])
                st.write(input_var) 
                
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                lb = LabelEncoder()
                scaler = StandardScaler()
                for i in input_var:
                    if input_var[i].dtypes == 'int' or input_var[i].dtypes == 'float':
                        input_var[[i]] = scaler.fit_transform(input_var[[i]])
                    else:
                        input_var[i] = lb.fit_transform(input_var[i])
                        
                # time.sleep(2)
                prediction = mod.predict(input_var)
                if prediction == 0:
                    st.error('Not qualified to have bank account')
                else:
                    st.balloons
                    st.success('You are qualified to have bank account')
                # st.write(prediction)
