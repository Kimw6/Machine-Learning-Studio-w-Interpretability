from local_components.containers import card_container
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.datasets import load_breast_cancer
from Models.Model_PCA import PCAModel
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from utilityFunctions import reset_for_new_dataset
import utilityFunctions as progress

st.set_page_config(layout="centered", page_title="Upload & Preprocess Data", page_icon="📁")
progress.render_sidebar(read_only=True)

# Function to load data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Set default session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'col_remove' not in st.session_state:
    st.session_state.col_remove = []
if 'norm_method' not in st.session_state:
    st.session_state.norm_method = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None

# List of normalization methods
NORM_TYPE = [None, 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer']

def estimate_memory(df):
    return sys.getsizeof(df) / (1024 * 1024)  # MB

def optimize_df_memory(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for i in range(0, len(numeric_cols), 1000):
        batch = numeric_cols[i:i+1000]
        for col in batch:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if len(df[col].unique()) / len(df[col]) < 0.5:
            df[col] = df[col].astype('category')
    return df

def upload_file():
    st.title("Upload & Preprocess Data")
    with st.expander("Upload Your Data", expanded=True):
        st.subheader("Upload Your Data for Preprocessing")
        with card_container(key="uploaded_data"):
            st.write("")
            upload_file = st.file_uploader('Upload your data here', type=['csv'])
            if upload_file is not None:
                st.write(f"Uploaded file size: {upload_file.size / (1024 * 1024):.2f} MB")
                if st.button('Upload'):
                    try:
                        temp_df = None
                        extension = upload_file.name.split('.')[-1].lower()
                        if extension == 'csv':
                            temp_df = pd.read_csv(upload_file)
                        elif extension == 'xlsx':
                            temp_df = pd.read_excel(upload_file)
                        elif extension == 'parquet':
                            temp_df = pd.read_parquet(upload_file)
                        elif extension == 'feather':
                            temp_df = pd.read_feather(upload_file)

                        temp_df = optimize_df_memory(temp_df)
                        mem_usage = estimate_memory(temp_df)

                        if mem_usage > 200:
                            st.session_state['memory_warning'] = True
                            st.error("Warning: This file may be too large for processing. Consider reducing its size.")
                            return
                        with st.container():
                            st.subheader("Overview of Uploaded Data")
                            with card_container(key="uploaded_data_preview"):
                                st.write("")
                                st.write('Data Uploaded Successfully')
                                st.write('***Data Shape***: ', temp_df.shape)
                                st.write(temp_df)
                                st.write('***Data Description***')
                                st.write(temp_df.describe())

                        temp_df = temp_df.dropna().reset_index(drop=True)

                        st.session_state['df'] = temp_df
                        reset_for_new_dataset(rerun=False, clear_cache=True)


                        st.session_state['histogram'] = None
                        
                        progress.set_done("qs_upload", True)
                    except Exception as e:
                        st.error('Error: {}'.format(e))


    st.write("""***Or Use the Breast Cancer Dataset***
        More info on [Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)""")

    if st.button('Load Breast Cancer Dataset'):
        try:
            temp_df = load_data()
            with st.expander("View Your Data", expanded=True):
                st.subheader("Overview of Uploaded Data")
                with card_container(key="uploaded_data_breast_cancer"):
                    st.write("")
                    st.write('Data Uploaded Successfully')
                    st.write('***Data Shape***: ', temp_df.shape)
                    st.write("***Overview of Data***")
                    st.write(temp_df)
                    st.write('***Data Description***')
                    st.write(temp_df.describe())

                    temp_df = temp_df.dropna().reset_index(drop=True)
                    st.session_state['df'] = temp_df

            reset_for_new_dataset(rerun=False, clear_cache=True)

            progress.set_done("qs_upload", True)
        except Exception as e:
            st.error('Error: {}'.format(e))

def apply_normalization(normalization_type):
    normalization_functions = {
        'None': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'RobustScaler': RobustScaler(),
        'QuantileTransformer': QuantileTransformer()
    }
    return normalization_functions[normalization_type]

def download_link():
    temp_df = st.session_state['df']
    with st.expander("View Your Cleaned Data", expanded=True):
        st.subheader("Overview of Data After Cleaning & Normalization")
        with card_container(key="cleaned_data"):
            st.write("")
            st.write('***Data Shape After Cleaning & Normalization***: ', temp_df.shape)
            st.write(temp_df)
            st.write(temp_df.describe())

            st.warning(""" If you need a copy of the cleaned data, please click the download button below.
                    Otherwise, you can proceed to the next step by selecting the model you want to use
                    on the left side of the screen.""")
            st.download_button('💾 Download Data', data=temp_df.to_csv(index=False), file_name='cleaned_data.csv', mime='text/csv')

def process_data():
    def on_click():
        if not st.session_state['temp_target']:
            st.error('Please select a target variable')
            return
        st.session_state['target'] = st.session_state['temp_target']
        progress.set_done("qs_target", True)

    st.warning("""***Note***: This will remove all rows with missing values. Currently does not handle
               non-numeric values. For the breast cancer dataset, target variable is 'target' """)
    cols = st.session_state['df'].columns
    with st.form(key='process_data'):
        with st.expander("Select columns to remove, target variable, and normalization method", expanded=True):
            st.subheader("Select columns to remove, target variable, and normalization method")
            with card_container(key="selection"):
                st.write("")
                st.multiselect('Select the columns to remove', cols, key='col_remove')
                st.selectbox(label='***Select the target variable***', options=cols, index=len(cols)-1, key='temp_target')
                st.session_state['norm_method'] = 'None'
                st.selectbox(label='***Select the normalization method***', options=NORM_TYPE, index=0, key='norm_method')
                st.form_submit_button('Process Data', on_click=on_click)

def normalize_delete():
    temp_df = st.session_state['df']
    target = st.session_state['target']

    with st.spinner('🔄 Processing data...'):
        if len(st.session_state['col_remove']) > 0:
            temp_df = temp_df.drop(st.session_state['col_remove'], axis=1)

        norm_type = apply_normalization(st.session_state['norm_method'])

        if norm_type is None:
            st.session_state['df'] = temp_df
        else:
            target_df = temp_df[target]
            temp_df = temp_df.drop(target, axis=1)
            cols = temp_df.columns
            try:
                temp_df = norm_type.fit_transform(temp_df)
                temp_df = pd.DataFrame(temp_df, columns=cols)
                temp_df = temp_df.dropna()
                temp_df[target] = target_df
                st.session_state['df'] = temp_df
            except Exception:
                st.session_state['df'] = temp_df
                st.error('Error: {}'.format('Your data contains non-numeric values. Please remove them and try again.'))
                return
    download_link()

# run
upload_file()

if st.session_state['df'] is not None:
    process_data()
    if st.session_state['target'] is not None:
        normalize_delete()