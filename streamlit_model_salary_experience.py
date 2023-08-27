

import numpy as np
import pickle as pk
import streamlit as st

loaded_model=pk.load(open('trained_model.sav','rb'))

def salary_prediction(input_data):
    input_experience_array=np.asarray(input_data)
    input_experience_array_new=input_experience_array.reshape(1,-1)
    experience_salary=loaded_model.predict(input_experience_array_new)
    import math
    experience_salary_new='Monthly Salary With Experience {1} Is {0} Rupees'.format(math.floor(experience_salary),input_data)
    annual_experience_salary='Annual Salary With Experience {1} Is {0} Rupees'.format(math.floor(12*experience_salary),input_data)
    return experience_salary_new,annual_experience_salary

def main():
    st.title('Salary Prediction Using Machine Learning')
    experience=st.number_input('Enter Work Experience Of Person')
    salary_with_experience=' '
    if st.button('Click Here To Get Expected Salary'):
        salary_with_experience=salary_prediction(experience)
    st.success(salary_with_experience)
    st.subheader('Exploratory Data Analysis Done And Machine Learning Model Deployed By "Anubhav Kumar Gupta"')
if __name__=='__main__':
    main()



