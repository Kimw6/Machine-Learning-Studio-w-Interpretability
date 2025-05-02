
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

# needs fix
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import streamlit_shadcn_ui as ui
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.special import softmax  # For converting decision functions to probabilities

from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance
from local_components import card_container
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, roc_curve, auc

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

    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    st.session_state['X_train'] = X_train
    st.session_state['y_train'] = y_test

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
    # Retrieve X_test and y_test from session state
    X_test = st.session_state.get('X_test')
    y_test = st.session_state.get('y_test')

    if X_test is None or y_test is None:
        st.warning("Test data not found. Please train the model first.")
        return

    # Display key metrics upfront
    st.write('#### Model Performance Metrics')
    cols = st.columns(4)

    with cols[0]:
        ui.card(
            title="Accuracy",
            content="{:.2f}%".format(model['accuracy'] * 100),
            description="Overall model accuracy",
            key="accuracy_card"
        ).render()
    with cols[1]:
        ui.card(
            title="Precision (Avg)",
            content="{:.2f}%".format(model['metrics_report']['macro avg']['precision'] * 100),
            description="Precision across classes",
            key="precision_card"
        ).render()
    with cols[2]:
        ui.card(
            title="Recall (Avg)",
            content="{:.2f}%".format(model['metrics_report']['macro avg']['recall'] * 100),
            description="Recall across classes",
            key="recall_card"
        ).render()
    with cols[3]:
        ui.card(
            title="F1-Score (Avg)",
            content="{:.2f}%".format(model['metrics_report']['macro avg']['f1-score'] * 100),
            description="F1-score across classes",
            key="f1_card"
        ).render()

    # Add an expandable section for the full classification report
    with st.expander("View Full Classification Report"):
        st.write('### Classification Report for', model['name_of_model'])
        report_df = pd.DataFrame(model['metrics_report']).transpose().reset_index()
        report_df.rename(columns={'index': 'Class'}, inplace=True)  # Rename the first column to 'Class'

        # Format the table for better readability
        report_df['precision'] = report_df['precision'].apply(lambda x: "{:.2f}".format(x) if not pd.isna(x) else "")
        report_df['recall'] = report_df['recall'].apply(lambda x: "{:.2f}".format(x) if not pd.isna(x) else "")
        report_df['f1-score'] = report_df['f1-score'].apply(lambda x: "{:.2f}".format(x) if not pd.isna(x) else "")
        report_df['support'] = report_df['support'].apply(lambda x: "{:.0f}".format(x) if not pd.isna(x) else "")

        # Display the table in a card container
        with card_container(key="classification_report"):
            ui.table(data=report_df, maxHeight=300)

    # Add an expandable section for the confusion matrix
    with st.expander("View Confusion Matrix"):
        st.write('### Confusion Matrix for', model['name_of_model'])
        with card_container(key="confusion_matrix"):
            fig, ax = plt.subplots()
            sns.heatmap(model['confusion_matrix'], annot=True, fmt='d', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

    # Check if the model supports ROC and Precision-Recall curves
    if hasattr(model['model'], "predict_proba") or hasattr(model['model'], "decision_function"):
        # Add an expandable section for the ROC curve
        with st.expander("View ROC Curve"):
            st.write('### ROC Curve for', model['name_of_model'])
            with card_container(key="roc_curve"):
                fig, ax = plt.subplots()
                RocCurveDisplay.from_estimator(model['model'], X_test, y_test, ax=ax)
                ax.set_title('ROC Curve')
                st.pyplot(fig)

        # Add an expandable section for the Precision-Recall curve
        with st.expander("View Precision-Recall Curve"):
            st.write('### Precision-Recall Curve for', model['name_of_model'])
            with card_container(key="precision_recall_curve"):
                fig, ax = plt.subplots()
                PrecisionRecallDisplay.from_estimator(model['model'], X_test, y_test, ax=ax)
                ax.set_title('Precision-Recall Curve')
                st.pyplot(fig)
    else:
        st.warning("ROC and Precision-Recall curves are not supported for this model.")

def reset_model(model_name):
    if model_name in st.session_state['models']:
        # Clear the model and its results
        st.session_state['models'][model_name] = None
        st.session_state['tabs'][model_name] = 'RESULTS'
        st.session_state['train'] = False  # Reset the training state
        st.success(f"Model '{model_name}' has been reset.")
    else:
        st.warning(f"No model found with the name '{model_name}'.")
# Source: https://discuss.streamlit.io/t/display-shap-diagrams-with-streamlit/1029/9
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def display_eda():
    # Display the select component to choose a histogram
    if 'histograms' in st.session_state and st.session_state['histograms']:
        with st.expander("View Feature Distributions", expanded=False):  # Collapsed by default
            features_list = list(st.session_state['histograms'].keys())
            selected_feature = st.selectbox("Select a feature", options=features_list, key="histogram_selector")
            st.markdown(f"**Selected Feature:** {selected_feature}")
            st.write("#### Features distributions")
            with card_container(key="eda"):
                # Display the selected histogram
                display_selected_histogram(selected_feature)

def display_selected_histogram(selected_feature):
    if 'histograms' in st.session_state and selected_feature in st.session_state['histograms']:
        with card_container(key=f"display_histogram_{selected_feature}"):  # Use a unique key for each card
            st.write(f"#### {selected_feature} distributions")  # Add a title for the histogram
            st.pyplot(st.session_state['histograms'][selected_feature])  # Display the selected histogram

def display_correlation_matrix():
    with st.expander("View Correlation Matrix", expanded=False):  # Collapsed by default
        with card_container(key="correlation_matrix"):
            f, ax = plt.subplots(figsize=(10, 6))
            corr = st.session_state['df'].corr()
            hm = sns.heatmap(round(corr,2), annot=False, ax=ax, cmap="Reds",fmt='',
                             linewidths=.05)
            f.subplots_adjust(top=0.93)
            t= f.suptitle('Attributes Correlation Heatmap', fontsize=14)
            st.pyplot(f)

def display_shap(model):
    st.write("### SHAP Diagrams")

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
        st.write(shap_values)
        max_features = min(100, len(X.columns))  # Limit to 20 features or the total number of features
        num_features = st.slider(
        "Select the number of features to remove",
        min_value=1,
        max_value=max_features,
        value=max_features,  # Default value
        key="shap_num_features1"
        )
        # Collapsible section for SHAP bar plot
        with st.expander("View SHAP Bar Plot", expanded=False):  # Collapsed by default
            st.write("#### SHAP Bar Plot")
            with card_container(key="shap_bar_plot"):
                fig, ax = plt.subplots()  # Create a figure and axis
                shap.summary_plot(shap_values, X, plot_type="bar", show=False)  # Create a bar plot
                st.pyplot(fig)  # Pass the figure explicitly
                plt.clf()

        # Collapsible section for SHAP summary plot
        with st.expander("View SHAP Summary Plot", expanded=False):  # Expanded by default
            # Add a slider to select the number of features in the summary plot
            max_features = min(100, len(X.columns))  # Limit to 20 features or the total number of features
            num_features = st.slider(
                "Select the number of features to display in the SHAP summary plot:",
                min_value=1,
                max_value=max_features,
                value=max_features,  # Default value
                key="shap_num_features"
            )

            st.write("#### SHAP Summary Plot")
            with card_container(key="shap_summary_plot"):
                fig, ax = plt.subplots()  # Create a figure and axis
                shap.summary_plot(shap_values, X, max_display=num_features, show=False)  # Limit the number of features
                st.pyplot(fig)  # Pass the figure explicitly

        # Collapsible section for SHAP force plot
        with st.expander("View SHAP Force Plot", expanded=False):  # Collapsed by default
            st.write("#### What are observations?")
            st.write("Each observation represents a single row in your dataset. "
                     "For example, if your dataset (breast cancer) contains information about patients, each observation corresponds to a specific patient.")

            # Add a dropdown to select an observation for the force plot
            default_observation_index = 0  # Set a default observation index

            observation_index = st.selectbox(
                "Select an observation for the SHAP force plot:",
                options=list(range(len(X))),  # Allow selection of any observation
                index=default_observation_index,  # Set the default selected observation
                format_func=lambda x: f"Observation {x + 1}",  # Display as "Observation 1", "Observation 2", etc.
                key="shap_observation_index"
            )

            st.write(f"#### Feature Values for Observation {observation_index + 1}")

            # Display the table in a card container
            with card_container(key="classification_report"):
                ui.table(data=X.iloc[observation_index:observation_index + 1], maxHeight=300)

            with card_container(key="shap_force_plot"):
                st.write(f"#### SHAP Force Plot for Observation {observation_index + 1}")
                # shap.initjs()  # Initialize JavaScript for SHAP
                force_plot = shap.force_plot(
                    explainer.expected_value,
                    shap_values[observation_index, :],  # Use the selected observation
                    X.iloc[observation_index, :]
                )
                st_shap(force_plot)  # Render the force plot using st_shap

    except Exception as e:
        st.error(f"An error occurred while calculating SHAP values: {e}")

def display_lime(model):
    st.write("### LIME Explanation for", model['name_of_model'])

    # Check if requirements are available
    if 'models' not in st.session_state or not st.session_state['models']:
        st.warning("No trained models found. Please train models first.")
        return

    if 'X_test' not in st.session_state or 'X_train' not in st.session_state:
        st.warning("Test/train data not found. Please run train model first.")
        return

    # Get the current active model
    try:
        # Handle case where model is stored as dictionary (common in Streamlit)
        if isinstance(model, dict):
            if 'model' in model:
                model = model['model']  # Extract actual model object
            else:
                st.error("Model dictionary doesn't contain 'model' key")
                return
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return

    # Prepare data
    X_test = st.session_state['X_test']
    X_train = st.session_state['X_train']
    y_test = st.session_state.get('y_test', None)
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else list(range(X_train.shape[1]))

    # Handle class names
    if hasattr(model, 'classes_'):
        class_names = [str(c) for c in model.classes_]
    elif 'y_train' in st.session_state:
        class_names = [str(c) for c in np.unique(st.session_state['y_train'])]
    else:
        class_names = ['Class 0', 'Class 1']  # Default binary

    # Create prediction function that works with all model types
    def predict_fn(x):
        try:
            # Handle pandas DataFrame if needed
            if hasattr(x, 'values'):
                x = x.values

            # Try predict_proba first (classification)
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(x)
                # Handle binary case where only positive class is returned
                if probas.shape[1] == 1:
                    return np.hstack([1-probas, probas])
                return probas

            # Try decision_function (SVMs)
            elif hasattr(model, 'decision_function'):
                dec = model.decision_function(x)
                if len(dec.shape) == 1:  # Binary case
                    return np.vstack([-dec, dec]).T / 2 + 0.5
                return softmax(dec)  # Multiclass

            # Fall back to predict
            preds = model.predict(x)
            if preds.ndim == 1:  # Needs to be 2D for LIME
                return preds.reshape(-1, 1)
            return preds

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            raise

    # Initialize explainer
    try:
        explainer = LimeTabularExplainer(
            training_data=X_train.values if hasattr(X_train, 'values') else X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification' if len(class_names) > 1 else 'regression',
            verbose=False,
            kernel_width=3
        )

        with st.expander("View LIME Explanation", expanded=False):
            # Let user select which instance to explain
            instance_idx = st.select_slider(
                "Select test instance to explain",
                options=list(range(len(X_test))),
                value=min(5, len(X_test)-1)
            )

            # Add the instance details display here
            st.write("### Selected Instance Details")
            if hasattr(X_test, 'iloc'):
                st.dataframe(X_test.iloc[instance_idx:instance_idx+1])
            else:
                st.write(pd.DataFrame([X_test[instance_idx]], columns=feature_names))

            # Show actual value if available
            if y_test is not None:
                actual_value = y_test.iloc[instance_idx] if hasattr(y_test, 'iloc') else y_test[instance_idx]
                st.write(f"**Actual Value:** {actual_value}")

            # Generate explanation
            explanation = explainer.explain_instance(
                X_test.iloc[instance_idx].values if hasattr(X_test, 'iloc') else X_test[instance_idx],
                predict_fn,
                num_features=10
            )

# In your display_lime function, modify the explanation section:

            with card_container(key="lime_explanation"):
                # Show the visualization
                fig = explanation.as_pyplot_figure()
                st.pyplot(fig)
                # Add interpretation guide
                st.write("""
                **How to read this explanation:**
                - **Class 1** means **Breast Cancer Present** (Malignant)
                - **Class 0** means **No Cancer** (Benign)
                - Green features support the prediction, red features oppose it
                """)
                # Enhanced actual value display
                if y_test is not None:
                    actual = y_test.iloc[instance_idx] if hasattr(y_test, 'iloc') else y_test[instance_idx]
                    diagnosis = "Malignant (Cancer Present)" if actual == 1 else "Benign (No Cancer)"
                    st.write(f"**Pathologist's Diagnosis (Ground Truth):** {actual} = {diagnosis}")

                # Enhanced prediction display
                prediction = predict_fn(X_test.iloc[instance_idx:instance_idx+1] if hasattr(X_test, 'iloc')
                                        else np.array([X_test[instance_idx]]))

                if len(prediction.shape) == 2 and prediction.shape[1] > 1:
                    st.write("#### Model's Prediction Confidence")
                    proba_df = pd.DataFrame(prediction, columns=["Benign (No Cancer)", "Malignant (Cancer)"])
                    st.dataframe(proba_df.style.format('{:.2%}'))

    except Exception as e:
        st.error(f"LIME explanation failed: {str(e)}")
        st.write("Common fixes:")
        st.write("- For SVM, ensure you used `probability=True`")
        st.write("- For neural networks, ensure they implement `predict_proba`")
        st.write("- For regression, ensure your target is continuous")
def display_pdp(model):
    st.write("### Displaying PDP")

    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.warning("No dataset found. Please upload data first.")
        return

    df = st.session_state['df']
    target = st.session_state['target']
    X = df.drop(target, axis=1)

    
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

# --------------------- TESTING -----------