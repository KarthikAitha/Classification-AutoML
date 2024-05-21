import streamlit as st
from pycaret.classification import setup, compare_models, pull, save_model
import pandas as pd

def main():
    st.title("Auto ML Web Application")

    # Upload data file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())


        st.subheader("Modelling")
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            setup(df, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)

            if st.button('Save Model'):
                save_model(best_model, 'best_model')
                st.success('Model saved successfully!')

if __name__ == "__main__":
    main()
