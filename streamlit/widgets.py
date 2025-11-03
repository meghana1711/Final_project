import streamlit as st

st.title("Streamlit text input")

name = st.text_input("Enter your name:")

age = st.slider("Select your age:",0,100,20)

st.write(f"Your age is {age}.")

options = ['x','y','z','w']
choice = st.selectbox('choose an option',options)
st.write(f"You selected {choice}")

if name:
    st.write(f"Hello, {name}")

data = {
    "Name": ['AB','CD','EF','GH'],
    "Age" :[25,28,30,32],
    "City":['CHN','DHL','BLR','HYB']
}
import pandas as pd
df1 = pd.DataFrame(data)
df1.to_csv("abc.csv")
st.write(df1)

upload_file = st.file_uploader("choose a file", type="csv")

if upload_file is not None:
    df=pd.read_csv(upload_file)
    st.write(df)