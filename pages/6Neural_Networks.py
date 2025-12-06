import streamlit as st
from Models.Model_NN import NeuralNetworkModel
import streamlit_shadcn_ui as ui
import utilityFunctions as progress

st.set_page_config(layout="centered", page_title="Neural Network for Classification", page_icon="🧠")
progress.render_sidebar(read_only=True)
st.title("Neural Network for Classification")

# defaults
default_session_state = {
    'df': None,
    'target': None,
    'trained_nn': False,
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize model container
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'Neural Network' not in st.session_state['models']:
    st.session_state['models']['Neural Network'] = None

# main flow
if st.session_state['df'] is not None and st.session_state['target'] is not None:
    model = NeuralNetworkModel()
    model.parameters()  # show form if not trained

    # If trained, show reset and results
    if st.session_state['models']['Neural Network']:

        model_data = st.session_state['models']['Neural Network']
        if model_data and model_data.get('trained'):
            progress.set_done("qs_train", True)
            if st.button("Reset Neural Network"):
                model.reset_model()
            model.display_results(model_data)

else:
    st.error("Oops, looks like your data is unavailable. \n\n Please try uploading your data again.", icon="⚠️")
    if ui.button("Re-upload Data", key="page_link_btn", variant="outline"):
        st.switch_page("pages/1Preprocess_Data.py")
