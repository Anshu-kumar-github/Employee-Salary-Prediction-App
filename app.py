

import streamlit as st
import joblib
import numpy as np

st.title("Salary Prediction App using Machine Learning")

st.divider()

st.write("This app estimates your salary based on your experience and job rating.")

years=st.number_input("Enter the Experience (in years)",value=1,step=1,min_value=0)
jobrate=st.number_input("Enter your Job rating",value=3.5,step=0.5,min_value=0.0)


x=[years,jobrate]

model=joblib.load("linearmodel.pkl")

st.divider()


predict=st.button("Press this for Salary prediction")

st.divider()

if predict:

    st.balloons()

    X1=np.array([x])

    prediction = model.predict(X1)

    st.write(f"Salary prediction is {prediction[0]:,.2f}")


else:
    "Please press the button for the prediction of your Salary"