from typing import Optional
import streamlit as st
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

MODEL_FLAGS = {
    'Linear Support Vector Machine': ['linear_model_trained', 'train'],
    'Logistic Regression':           ['linear_model_trained', 'train'],

    'Decision Tree Classifier':      ['decision_tree_trained', 'train'],

    'Random Forest Classifier':      ['ensemble_trained', 'train'],
    'Gradient Boosting Classifier':  ['ensemble_trained', 'train'],
    'AdaBoost Classifier':           ['ensemble_trained', 'train'],

    'KMeans Clustering':             ['clustered', 'train'],
    'Dimensionality Reduction-PCA':  ['clustered', 'train'],

    'Neural Network':                ['trained_nn'],  
}

def _clear_gating_flags():
    """Turn off every UI gating flag so forms reappear after resets."""
    for flags in MODEL_FLAGS.values():
        for f in flags:
            st.session_state[f] = False

def ensure_defaults():
    if 'models' not in st.session_state:
        st.session_state['models'] = {}
    if 'tabs' not in st.session_state:
        st.session_state['tabs'] = {}

def _clear_training_artifacts():
    # shared artifacts (leave alone for per-model resets)
    for k in [
        'X_train','X_test','y_train','y_test',
        'accuracy','metrics_report','cm','temp_name'
    ]:
        st.session_state.pop(k, None)

# ---------- public resets ----------

def reset_model(model_name: str, *, clear_cache: bool = False, rerun: bool = True):
    """
    Reset only one model:
      - remove its entry from st.session_state['models']
      - reset its tab to 'RESULTS'
      - set the model's gating flags to False so its UI (form) appears again
      - DO NOT clear shared training artifacts (so other models keep working)
    """
    ensure_defaults()

    if clear_cache:
        try: st.cache_data.clear()
        except Exception: pass
        try: st.cache_resource.clear()
        except Exception: pass

    st.session_state['models'].pop(model_name, None)
    st.session_state['tabs'][model_name] = 'RESULTS'

    for flag in MODEL_FLAGS.get(model_name, []):
        st.session_state[flag] = False

    # per-model extras
    if model_name in ('KMeans Clustering',):
        st.session_state.pop('df_clustered', None)

    if rerun:
        st.rerun()

def reset_for_new_dataset(df: Optional[pd.DataFrame] = None, *, clear_cache: bool = False, rerun: bool = True):
    ensure_defaults()

    if df is not None:
        st.session_state['df'] = df

    st.session_state['models'].clear()
    st.session_state['tabs'].clear()

    for k in ['histograms','data_summary','histogram','df_clustered']:
        st.session_state.pop(k, None)

    _clear_training_artifacts()
    _clear_gating_flags()      # <-- add this

    if clear_cache:
        try: st.cache_data.clear()
        except Exception: pass
        try: st.cache_resource.clear()
        except Exception: pass

    if rerun:
        st.rerun()

# ---------- training / eval / display ----------

def train_model(model, feature_importance: bool = False, decision_tree: bool = False):
    df = st.session_state['df']
    target = st.session_state['target']
    test_size = st.session_state['test_size']
    random_state = st.session_state['random_state']

    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    st.session_state['X_train'] = X_train
    st.session_state['y_train'] = y_train

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        handle_errors(e)
        return

    y_pred = evaluate_model(model, X_test, y_test)

    if feature_importance:
        st.write('Feature Importance')
    if decision_tree:
        permutation_importance(model, X_test, y_test)

    return (model, y_pred)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    metrics_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    save_model_results(model, st.session_state['temp_name'], accuracy, metrics_report, cm)
    return y_pred

def confusion_matrix(y_test, y_pred):
    return metrics.confusion_matrix(y_test, y_pred)

def save_model_results(model, model_name, accuracy, metrics_report, cm):
    st.write("")
    st.session_state['models'][model_name] = {
        'model': model,
        'accuracy': accuracy,
        'metrics_report': metrics_report,
        'confusion_matrix': cm,
        'name_of_model': model_name,
        'X_test': st.session_state.get('X_test'),
        'y_test': st.session_state.get('y_test'),
    }

def handle_errors(e):
    st.error(f"Something went wrong. An error occurred: {e}")
