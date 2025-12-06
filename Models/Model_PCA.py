import streamlit as st
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import utilityFunctions as progress

class PCAModel:
    def __init__(self):
        self.params = {
            'n_components': None,
            'copy': True,
            'whiten': False,
            'svd_solver': 'auto',
            'tol': 0.0,
            'iterated_power': 'auto',
            'n_oversamples': 10,
            'power_iteration_normalizer': 'auto',
            'random_state': None,
        }
    def parameters(self):
        if not st.session_state['clustered']:
            st.write("""***Please adjust the parameters below to configure your PCA model.*** More 
                    info on [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)""")
            
            with st.form(key='pca_form'):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.number_input('n_components', min_value=1, value = 2,  max_value=30, key='n_components')
                    st.selectbox('svd_solver', ['auto', 'full', 'arpack', 'randomized'], key='svd_solver')
                    st.number_input('tol', step=0.01, key='tol')
                    st.selectbox('copy', [True, False], key='copy', index=0)
                    st.number_input('n_oversamples', min_value=1, key='n_oversamples')
                
                with col2:
                    
                    st.selectbox('iterated_power', ['auto', 'exact', 'randomized'], key='iterated_power')
                    st.selectbox('power_iteration_normalizer', ['auto', 'QR', 'LU'], key='power_iteration_normalizer')
                    st.number_input('random_state', min_value=1, value = 42, key='random_state')
                    st.checkbox('Whiten', key='whiten')
                    
                st.form_submit_button('Apply PCA', on_click=self.on_click)

    def on_click(self):
        progress.set_done("qs_train", True)
        self.params = {key: st.session_state[key] for key in self.params.keys()}
        try:
            pca = PCA(**self.params)
            self.train_model(pca)
            st.session_state['clustered'] = True
        except Exception as e:
            st.error(e)
    
    def train_model(self, pca):
        X = st.session_state['df'].drop(st.session_state['target'], axis=1)

        pca.fit(X)
        components = pca.transform(X)

        self.save_model_results(pca, st.session_state['temp_name'], X, components)

    def save_model_results(self, model, model_name, X, components):
        st.session_state['models'][model_name] = {
            'model': model,
            'X': X,
            'components': components,
            'name_of_model': model_name,
            'clustered': True
        }

    
