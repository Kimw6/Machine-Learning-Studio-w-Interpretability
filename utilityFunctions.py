
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
# needs fix
import streamlit.components.v1 as components
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance
import shap

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# This function trains the model and displays the accuracy,
# classification report, and confusion matrix
def train_model(model, feature_importance=False, decision_tree=False):
    df = st.session_state['df']
    target = st.session_state['target']
    test_size = st.session_state['test_size']
    random_state = st.session_state['random_state']
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

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
    #     render_tree(model, X, y)
    return (model, y_pred)

# This function evaluates the model and displays the accuracy, 
# classification report, and confusion matrix
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    metrics_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    # for row in metrics_report:
    #     st.write(row, metrics_report[row])
    cm = confusion_matrix(y_test, y_pred)

    save_model_results(model, st.session_state['temp_name'], accuracy, metrics_report, cm)
    return y_pred

# Confusion Matrix creates a heatmap of the confusion matrix
def confusion_matrix(y_test, y_pred):
    cm = metrics.confusion_matrix(y_test, y_pred)
    return cm

def save_model_results(model, model_name, accuracy, metrics_report, cm):
    st.session_state['models'][model_name] = {
        'model': model,
        'accuracy': accuracy,
        'metrics_report': metrics_report,
        'confusion_matrix': cm,
        'name_of_model': model_name,
    }

def display_results(model):
    st.write('***Accuracy: {:.2f}***'.format(model['accuracy']))
    st.write('Classification Report for', model['name_of_model'])

    data = pd.DataFrame(model['metrics_report']).transpose()
    st.write(data)
    fig, ax = plt.subplots()
    sns.heatmap(model['confusion_matrix'] , annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def reset_model(model_name):
    """
    Reset all data related to a specific model.

    Parameters:
        model_name: The name of the model to reset (e.g., 'Linear Support Vector Machine').
    """
    if model_name in st.session_state['models']:
        # Clear the model and its results
        st.session_state['models'][model_name] = None
        st.session_state['tabs'][model_name] = 'RESULTS'  # Reset the tab to default
        st.success(f"Model '{model_name}' has been reset.")
    else:
        st.warning(f"No model found with the name '{model_name}'.")

def st_shap(plot, height=None):
    """
    Render a SHAP plot in Streamlit.

    Parameters:
        plot: A SHAP plot object (e.g., force_plot, dependence_plot).
        height: The height of the plot in pixels (optional).
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def display_EDA(df, precompute=False):
    pass
def display_shap(model):
    st.write("### SHAP Values")

    # Ensure the model and data are available
    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.warning("No dataset found. Please upload data first.")
        return

    df = st.session_state['df']
    target = st.session_state['target']
    X = df.drop(target, axis=1)

    try:
        # Check if the model is tree-based
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        # Check if the model is linear
        elif hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        # For other models, use KernelExplainer
        else:
            explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X)

        # Plot SHAP summary
        st.write("#### SHAP Summary Plot")
        fig, ax = plt.subplots()  # Create a figure and axis
        shap.summary_plot(shap_values, X, show=False)  # Disable auto-show
        st.pyplot(fig)  # Pass the figure explicitly

        # Plot SHAP force plot for the first observation
        st.write("#### SHAP Force Plot for First Observation")
        shap.initjs()  # Initialize JavaScript for SHAP
        force_plot = shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
        st_shap(force_plot)  # Render the force plot using st_shap

    except Exception as e:
        st.error(f"An error occurred while calculating SHAP values: {e}")
def display_lime(model):
    st.write("Displaying LIME")
    st.write("Displaying LIME")
def display_pdp(model):
    st.write("Displaying PDP")
    st.write("Displaying PDP")
def display_pfi(model):
    st.write("Displaying PFI")
    st.write("Displaying PFI")


# This function calculates the permutation importance of each feature
def perm_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, random_state=42)
    st.write(result.importances_mean)

# This need needs to be fixed. Future work to render the decision tree
def render_tree(model, columns, target):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, feature_names=columns, class_names=[str(t) for t in target], filled=True, ax=ax)
    st.pyplot(fig)

def handle_errors(e):
    st.warning('Something went wrong. Check your data set and try again. May contain non-numeric values.')

