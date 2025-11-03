import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_name = load_data()

model =RandomForestClassifier()
model.fit(df.iloc[:,:-1], df['species'])

st.sidebar.title("Input features")
s_length = st.sidebar.slider("sepal length", float(df["sepal length (cm)"].min()),  float(df["sepal length (cm)"].max()))
s_width = st.sidebar.slider("sepal width", float(df["sepal width (cm)"].min()),  float(df["sepal width (cm)"].max()))
p_length = st.sidebar.slider("petal length", float(df["petal length (cm)"].min()),  float(df["petal length (cm)"].max()))
p_width = st.sidebar.slider("petal width", float(df["petal width (cm)"].min()),  float(df["petal width (cm)"].max()))

input_data = [[s_length,s_width,p_length,p_width]]

pred = model.predict(input_data)
predicted = target_name[pred[0]]

st.write("Prediction")
st.write(f"The predicted spieces is: {predicted}")