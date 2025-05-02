from utilityFunctions import display_eda, display_correlation_matrix
import streamlit as st
import streamlit_shadcn_ui as ui

st.subheader('Quick Explatory Data Analysis (EDA) and Visualization')

if st.session_state['df'] is not None and st.session_state['target'] is not None:
    display_eda()
    display_correlation_matrix()
else:
    st.error("Oops, looks like your data is unavailable. \n\n Please try uploading you data again.", icon="⚠️")
    # Going to the previous tab
    if ui.button("Re-upload Data", key="page_link_btn", variant="outline"):
        st.switch_page("pages/1Preprocess_Data.py")  # Adjust path as needed

