import streamlit as st
import pandas as pd
import numpy as np

st.title("Hello Streamlit!")
st.write("✅ If you see this, it works.")

df = pd.DataFrame({
    'first': [1, 2, 3],
    'second': [10, 20, 30]
})

## Display the dataframe
st.write("Here is the dataframe")
st.write(df)


## create a line chart
chart_data = pd.DataFrame(
    np.random.randn(20, 3), columns=['a', 'b', 'c']
)

# FIX: Pass the chart_data DataFrame to the function
st.line_chart(chart_data)