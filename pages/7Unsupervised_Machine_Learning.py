import streamlit as st
from Models.Model_KMeans import KMeansClusteringModel
from Models.Model_PCA import PCAModel
from utilityFunctions import reset_model
import utilityFunctions as progress
import streamlit_shadcn_ui as ui
from local_components import card_container
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(layout="centered", page_title="Unsupervised Machine Learning", page_icon="🤖")
progress.render_sidebar(read_only=True)
st.title("Unsupervised Machine Learning")

default_session_state = {
    'df': None,
    'target': None,
    'model_type': None,
    'train' : False,
    'k_means_trained': False,
    'df_clustered': None,
    'clustered': None,
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

model_type = ['KMeans Clustering', 'Dimensionality Reduction-PCA']

if 'models' not in st.session_state:
    st.session_state['models'] = {}
for model in model_type:
    if model not in st.session_state['models']:
        st.session_state['models'][model] = None
if 'tabs' not in st.session_state:
    st.session_state['tabs'] = {model: 'RESULTS' for model in model_type}

def pca_sample_data(X, components):
    with st.expander("View PCA Sample Data", expanded=False):
        st.write('### PCA Sample Data')
        st.write("***Original shape:*** ", X.shape)
        st.write("***Transformed shape:***", components.shape)
        with card_container(key="pca_sample_data"):
            df = pd.DataFrame(components)
            target = st.session_state['target']
            df[target] = st.session_state['df'][target]
            st.write("")
            st.write(df.head(10))
        st.download_button('Download Data', data=df.to_csv(index=False),
                           file_name='transformed_data.csv', mime='text/csv')

def pca_scatter_plot(components):
    with st.expander("View Principal Component Analysis (PCA) Scatter Plot"):
        with card_container(key="pca_scatter_plot"):
            st.write("")
            st.write('### Principal Component Analysis Scatter Plot')
            target = st.session_state['target']
            y = st.session_state['df'][target]
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(components[:, 0], components[:, 1], c=y, cmap='viridis', alpha=0.8, label='Data Points')
            unique_classes = y.unique()
            centers = np.vstack([components[(y == cls)].mean(axis=0) for cls in unique_classes])
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers',
                        edgecolors='black', linewidths=1.5)
            for i, cls in enumerate(unique_classes):
                plt.text(centers[i, 0], centers[i, 1], f'Class {cls}', fontsize=10, ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3))
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('PCA Scatter Plot with Class Centers')
            plt.colorbar(scatter, label=target)
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

def kmeans_scatter_plot(df_clustered):
    with st.expander("View KMeans Clustering Scatter Plot", expanded=False):
        with card_container(key="kmeans_scatter_plot"):
            st.write("")
            st.write('### KMeans Clustering Scatter Plot')
            labels = df_clustered['cluster_labels']
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(df_clustered.iloc[:, 0], df_clustered.iloc[:, 1], c=labels, cmap='viridis', alpha=0.8)
            centers = df_clustered.groupby('cluster_labels').mean().values
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers',
                        edgecolors='black', linewidths=1.5)
            plt.xlabel('Feature 1'); plt.ylabel('Feature 2'); plt.title('KMeans Clustering with Centers')
            plt.colorbar(scatter, label='Cluster Labels'); plt.legend(); plt.grid(True)
            st.pyplot(plt)

def kmeans_sample_clustered_data(df_clustered):
    with st.expander("View KMeans Sample Clustered Data", expanded=False):
        st.write('### KMeans Sample Clustered Data')
        with card_container(key="kmeans_sample_clustered_data"):
            st.write("")
            st.write(df_clustered.head(10))
        st.download_button('Download Clustered Data', data=df_clustered.to_csv(index=False),
                           file_name='clustered_data.csv', mime='text/csv')

def display_results(model):
    if model["name_of_model"] == "Dimensionality Reduction-PCA":
        st.write("Principal Component Analysis (PCA) reduces dimensionality while preserving variance.")
        X = model["X"]; components = model["components"]
        pca_sample_data(X, components)
        pca_scatter_plot(components)
    if model["name_of_model"] == "KMeans Clustering":
        st.write("KMeans partitions data into K clusters based on similarity.")
        df_clustered = model["df_clustered"]
        kmeans_scatter_plot(df_clustered)
        kmeans_sample_clustered_data(df_clustered)

def handle_results(model_name):
    progress.set_done("qs_interpret", True)
    model_data = st.session_state['models'][model_name]
    if model_data.get('clustered'):
        display_results(model_data)

for model_name in model_type:
    if st.session_state['models'][model_name] is not None:
        if st.button(f"Reset {model_name}"):
            reset_model(model_name)  # centralized; also flips 'clustered' False
        handle_results(model_name)

def model_selection():
    def on_click():
        if st.session_state.get('temp_linear_model') is None:
            st.warning('Please select a model.', icon="⚠️"); return
        st.session_state['model_type'] = st.session_state['temp_linear_model']
        st.session_state['train'] = True

    if not st.session_state.get('train', False) and not st.session_state.get('clustered', False):
        with st.form(key='linear_model_multi'):
            st.selectbox("Select Model for Unsupervised Learning", model_type, key='temp_linear_model')
            st.form_submit_button('Select', on_click=on_click)

if st.session_state['df'] is not None  and st.session_state['target'] is not None:
    model_selection()
else:
    st.error("Oops, looks like your data is unavailable. \n\n Please try uploading you data again.", icon="⚠️")
    if ui.button("Re-upload Data", key="page_link_btn", variant="outline"):
        st.switch_page("pages/1Preprocess_Data.py")

if st.session_state['train']:
    model_type_sel = st.session_state['model_type']
    if model_type_sel == 'KMeans Clustering':
        try:
            model = KMeansClusteringModel()
            model.parameters()
            st.session_state['temp_name'] = model_type_sel
        except Exception as e:
            st.error(e)
        finally:
            st.session_state['train'] = False
    if model_type_sel == 'Dimensionality Reduction-PCA':
        try:
            model = PCAModel()
            model.parameters()
            st.session_state['temp_name'] = model_type_sel
        except Exception as e:
            st.error(e)
        finally:
            st.session_state['train'] = False
