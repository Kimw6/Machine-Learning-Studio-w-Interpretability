import streamlit as st
from sklearn.cluster import KMeans
import utilityFunctions as progress
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

BINARY = [True, False]
class KMeansClusteringModel:
    def __init__(self):
        self.params = {
            'n_clusters': 1,
            'init': 'k-means++',
            'n_init': 10,
            'max_iter': 300,
            'tol': 0.0001,
            'random_state': 42,
            'copy_x': True,
            'algorithm': 'lloyd',
        }
    
    def parameters(self):
        if not st.session_state['clustered']:
            st.write("""Please adjust the parameters below to configure your K-Means clustering model. 
                    ***Note that number of clusters set to number of classes avaible in target.***
                    More info on [K-Means Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)""")
            num_clusters = len(st.session_state['df'][st.session_state['target']].unique())
            with st.form(key='kmeans_clustering_form'):
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input('n_clusters', value=num_clusters, step=1, min_value=1, key='n_clusters')
                    st.selectbox('init', ['k-means++', 'random'], key='init')
                    st.number_input('n_init', value=10, step=1, min_value=1, key='n_init')
                    st.number_input('max_iter', value=300, step=10, min_value=1, key='max_iter')
                    st.selectbox('copy_x', BINARY, key='copy_x', index=0)
                
                with col2:
                    st.number_input('tol', value=0.0001, step=0.0001, min_value=0.0, key='tol')
                    st.selectbox('algorithm', ['lloyd', 'elkan'], key='algorithm')
                    #st.text("Optional: You can set a random seed for reproducibility:")
                    st.number_input('random_state', min_value=1, value=42, key='random_state')
                    
                st.form_submit_button('Train Model', on_click=self.on_click)

    def on_click(self):
        progress.set_done("qs_train", True)
        self.params = {key: st.session_state[key] for key in self.params.keys()}
        try:
            model = KMeans(**self.params)
            self.train_model(model)
            st.session_state['clustered'] = True
        except Exception as e:
            st.error(e)

    def train_model(self, kmeans):
        df_copy = st.session_state['df'].drop(st.session_state['target'], axis=1)

        kmeans.fit(df_copy)
        labels = kmeans.labels_

        df_clustered = st.session_state['df'].copy()
        df_clustered['cluster_labels'] = labels
        st.session_state['df_clustered'] = df_clustered

        target = np.array(st.session_state['df'][st.session_state['target']])

        num_classes = len(st.session_state['df'][st.session_state['target']].unique())

        if num_classes == self.params['n_clusters']:
            self.save_model_results(kmeans, st.session_state['temp_name'], df_clustered,)


    def save_model_results(self, model, model_name, df_clustered):
        st.session_state['models'][model_name] = {
            'model': model,
            'df_clustered': df_clustered,
            'name_of_model': model_name,
            'clustered': True
        }
