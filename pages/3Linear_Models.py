import streamlit as st
import streamlit_shadcn_ui as ui
from utilityFunctions import display_results, display_shap, display_lime, display_pdp, display_pfi, reset_model
from Model_Linear import LinearSupportVectorMachineModel, LogisticRegressionModel
import streamlit_shadcn_ui as ui


st.subheader('Linear Models for Classification')

default_session_state = {
    'df': None,
    'target': None,
    'linear_model': None,
    'test_size': None,
    'random_state': None,
    'train': None,
    'X_test': None,
    'y_test': None,
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

model_type = ['Linear Support Vector Machine', 'Logistic Regression']

if 'models' not in st.session_state:
    st.session_state['models'] = {}

for model in model_type:
    if model not in st.session_state['models']:
        st.session_state['models'][model] = None

if 'tabs' not in st.session_state:
    st.session_state['tabs'] = {model: 'RESULTS' for model in model_type}

if st.session_state['models'][model_type[0]] is not None:
    if st.button("Reset Model"):
        reset_model(model_type[0])
        st.session_state['train'] = False
        st.rerun()

    selected_tab = ui.tabs(
        options=['RESULTS', 'SHAP', 'LIME', 'PDP', 'PFI', 'Compare Models'],
        default_value=st.session_state['tabs'][model_type[0]],
        key=f"tabs_{model_type[0]}"
    )
    st.session_state['tabs'][model_type[0]] = selected_tab

    # Display content based on the selected tab
    if selected_tab == 'RESULTS':
        display_results(st.session_state['models'][model_type[0]])
    elif selected_tab == 'SHAP':
        display_shap(st.session_state['models'][model_type[0]]['model'])
    elif selected_tab == 'LIME':
        display_lime(st.session_state['models'][model_type[0]])
    elif selected_tab == 'PDP':
        display_pdp(st.session_state['models'][model_type[0]])
    # elif selected_tab == 'PFI':
    #     display_pfi(st.session_state['models'][model_type[0]])
    # elif selected_tab == 'Compare_Models':


elif st.session_state['models'][model_type[1]] is not None:
    if st.button("Reset Model"):
        reset_model(model_type[1])
        st.rerun()
    display_results(st.session_state['models'][model_type[1]])
    selected_tab = ui.tabs(
        options=['SHAP', 'LIME', 'PDP', 'PFI'],
        default_value=st.session_state['tabs'][model_type[1]],
        key=f"tabs_{model_type[1]}"  # Unique key for this model type
    )
    st.session_state['tabs'][model_type[1]] = selected_tab  # Update session state

    # Display content based on the selected tab
    if selected_tab == 'RESULTS':
        display_results(st.session_state['models'][model_type[1]])
    elif selected_tab == 'SHAP':
        display_shap(st.session_state['models'][model_type[1]]['model'])
    elif selected_tab == 'LIME':
        display_lime(st.session_state['models'][model_type[1]])
    # elif selected_tab == 'PDP':
    #     display_pdp(st.session_state['models'][model_type[1]])
    # elif selected_tab == 'PFI':
    #     display_pfi(st.session_state['models'][model_type[1]])
    # elif selected_tab == 'Compare Models':
    #     st.write("### Model Comparison")
    #
    #     if not st.session_state['trained_models']:
    #         st.warning("No trained models found. Please train at least one model.")
    #     else:
    #         # Display metrics for all trained models
    #         for model_name, model_data in st.session_state['trained_models'].items():
    #             st.write(f"#### {model_name}")
    #             display_results(model_data)
def linear_model():
    def on_click():
        if st.session_state['temp_linear_model'] is None:
            st.warning('Please select a model.', icon="⚠️")
            return
        if st.session_state['temp_test_size'] is None:
            st.warning('Please select a test size.', icon="⚠️")
            return
        if st.session_state['rand_state'] is None:
            st.warning('Please select a random state.', icon="⚠️")
            return
        st.session_state['linear_model'] = st.session_state['temp_linear_model']
        st.session_state['test_size'] = st.session_state['temp_test_size']
        st.session_state['random_state'] = st.session_state['rand_state']
        st.session_state['train'] = True

    num_classes = len(st.session_state['df'][st.session_state['target']].unique())

    if not st.session_state['train']:
        with st.form(key='linear_model_multi'):
            st.write('Number of Classes: ', num_classes)
            st.selectbox("Choose a Model for Classification", model_type, key='temp_linear_model')
            st.number_input("Choose the Test Data Size Ratio", key='temp_test_size',
                            value=0.2, step=0.05, min_value=0.05, max_value=0.9)
            st.number_input("Choose a Random State", key='rand_state',
                            value=42, step=1, min_value=0, max_value=1000)
            st.form_submit_button('Select', on_click=on_click)

if st.session_state['df'] is not None and st.session_state['target'] is not None:
    linear_model()
else:
    st.error("Oops, looks like your data is unavailable. \n\n Please try uploading you data again.", icon="⚠️")
    # Going to the previous tab
    if ui.button("Re-upload Data", key="page_link_btn", variant="outline"):
        st.switch_page("pages/1Preprocess_Data.py")  # Adjust path as needed

if st.session_state['train']:
    model_type = st.session_state['linear_model']
    if model_type == 'Linear Support Vector Machine':
        model = LinearSupportVectorMachineModel()
        model.parameters()
        st.session_state['temp_name'] = model_type

    if model_type == 'Logistic Regression':
        model = LogisticRegressionModel()
        model.parameters()

        st.session_state['temp_name'] = model_type

