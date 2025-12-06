import streamlit as st
from utilityFunctions import display_results, display_shap, display_lime, display_pdp, display_pfi, reset_model
import utilityFunctions as progress
from Models.Model_DecisionTree import DecisionTreeClassifierModel
import streamlit_shadcn_ui as ui

st.set_page_config(layout="centered", page_title="Decision Trees Models for Classification", page_icon="🌳")
progress.render_sidebar(read_only=True)
st.title('Decision Trees Models for Classification')

default_session_state = {
    'df': None,
    'target': None,
    'linear_model': None,
    'test_size': None,
    'random_state': None,
    'train' : False,
    'decision_tree_trained': False,
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

model_type = ['Decision Tree Classifier']

if 'models' not in st.session_state:
    st.session_state['models'] = {}
for model in model_type:
    if model not in st.session_state['models']:
        st.session_state['models'][model] = None
if 'tabs' not in st.session_state:
    st.session_state['tabs'] = {model: 'RESULTS' for model in model_type}

def handle_results(model_name):
    if model_name not in st.session_state['tabs']:
        st.session_state['tabs'][model_name] = 'RESULTS'
    selected_tab = ui.tabs(
        options=['RESULTS', 'SHAP', 'LIME', 'PDP', 'PFI'],
        default_value=st.session_state['tabs'][model_name],
        key=f"tabs_{model_name}"
    )
    st.session_state['tabs'][model_name] = selected_tab

    model_data = st.session_state['models'][model_name]

    if model_data:
        progress.set_done("qs_train", True)
    
    if selected_tab == 'RESULTS':
        display_results(model_data)
        progress.set_done("qs_interpret", True)
    elif selected_tab == 'SHAP':
        display_shap(model_data)
    elif selected_tab == 'LIME':
        display_lime(model_data)
    elif selected_tab == 'PDP':
        display_pdp(model_data)
    elif selected_tab == 'PFI':
        display_pfi(model_data)
        
for model_name in model_type:
    if st.session_state['models'][model_name] is not None:
        if st.button(f"Reset {model_name}"):
            reset_model(model_name)
        handle_results(model_name)

def model_selection():
    def on_click():
        st.session_state['accuracy'] = None
        st.session_state['metrics_report'] = None
        st.session_state['cm'] = None

        if st.session_state.get('temp_linear_model') is None:
            st.warning('Please select a model.', icon="⚠️"); return
        if st.session_state.get('temp_test_size') is None:
            st.warning('Please select a test size.', icon="⚠️"); return
        if st.session_state.get('rand_state') is None:
            st.warning('Please select a random state.', icon="⚠️"); return

        st.session_state['linear_model'] = st.session_state['temp_linear_model']
        st.session_state['test_size'] = st.session_state['temp_test_size']
        st.session_state['random_state'] = st.session_state['rand_state']
        st.session_state['train'] = True

    num_classes = len(st.session_state['df'][st.session_state['target']].unique())

    if not st.session_state.get('train', False) and not st.session_state.get('decision_tree_trained', False):
        with st.form(key='linear_model_multi'):
            st.write('Number of Classes: ', num_classes)
            st.selectbox("Choose a Model for Classification", model_type, key='temp_linear_model')
            st.number_input("Choose the Test Data Size Ratio", key='temp_test_size',
                            value=0.2, step=0.05, min_value=0.05, max_value=0.9)
            st.number_input("Choose a Random State", key='rand_state',
                            value=42, step=1, min_value=0, max_value=1000)
            st.form_submit_button('Select', on_click=on_click)

if st.session_state['df'] is not None and st.session_state['target'] is not None:
    model_selection()
else:
    st.error("Oops, looks like your data is unavailable. \n\n Please try uploading you data again.", icon="⚠️")
    if ui.button("Re-upload Data", key="page_link_btn", variant="outline"):
        st.switch_page("pages/1Preprocess_Data.py")

if st.session_state['train']:
    model_type_sel = st.session_state['linear_model']
    if model_type_sel == 'Decision Tree Classifier':
        model = DecisionTreeClassifierModel()
        model.parameters()
        st.session_state['temp_name'] = model_type_sel
    st.session_state['train'] = False
