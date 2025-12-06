import streamlit as st
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.special import softmax
from local_components import card_container

def _resolve_model(model):
    """Accept either the session 'model dict' or a raw estimator."""
    if isinstance(model, dict):
        if "model" not in model:
            raise ValueError("Model dictionary doesn't contain 'model' key")
        return model["model"]
    return model

def _get_lime_inputs(model):
    """
    Pulls X_train, X_test, y_test, feature_names, class_names
    from session state and the estimator.
    Raises ValueError with a message if something is missing.
    """
    if "models" not in st.session_state or not st.session_state["models"]:
        raise ValueError("No trained models found. Please train models first.")

    if "X_test" not in st.session_state or "X_train" not in st.session_state:
        raise ValueError("Test/train data not found. Please run train model first.")

    est = _resolve_model(model)

    X_test = st.session_state["X_test"]
    X_train = st.session_state["X_train"]
    y_test = st.session_state.get("y_test", None)

    feature_names = (
        X_train.columns.tolist()
        if hasattr(X_train, "columns")
        else list(range(X_train.shape[1]))
    )

    # figure out class names
    if hasattr(est, "classes_"):
        class_names = [str(c) for c in est.classes_]
    elif "y_train" in st.session_state:
        class_names = [str(c) for c in np.unique(st.session_state["y_train"])]
    else:
        class_names = ["Class 0", "Class 1"]

    return est, X_train, X_test, y_test, feature_names, class_names

def _make_predict_fn(est):
    """Wrap estimator into the function shape LIME expects."""
    def predict_fn(x):
        try:
            if hasattr(x, "values"):
                x = x.values

            if hasattr(est, "predict_proba"):
                probas = est.predict_proba(x)
                if probas.shape[1] == 1:
                    return np.hstack([1 - probas, probas])
                return probas

            if hasattr(est, "decision_function"):
                dec = est.decision_function(x)
                if len(dec.shape) == 1:
                    return np.vstack([-dec, dec]).T / 2 + 0.5
                return softmax(dec, axis=1)

            preds = est.predict(x)
            if preds.ndim == 1:
                return preds.reshape(-1, 1)
            return preds
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            raise

    return predict_fn

def compute_lime_explanation(model, instance_idx: int, num_features: int = 10):
    """
    Pure-ish LIME pipeline for a single instance.

    Returns:
        explanation (LimeExplanation),
        pred_proba (np.ndarray),
        predicted_class (int),
        actual_value (optional),
        X_row (1-row DataFrame for display),
        class_names (list[str])
    """
    est, X_train, X_test, y_test, feature_names, class_names = _get_lime_inputs(model)
    predict_fn = _make_predict_fn(est)

    explainer = LimeTabularExplainer(
        training_data=X_train.values if hasattr(X_train, "values") else X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification" if len(class_names) > 1 else "regression",
        verbose=False,
        kernel_width=3,
    )

    # grab row
    if hasattr(X_test, "iloc"):
        row = X_test.iloc[instance_idx]
        row_batch = X_test.iloc[instance_idx : instance_idx + 1]
    else:
        row = X_test[instance_idx]
        row_batch = np.array([X_test[instance_idx]])

    explanation = explainer.explain_instance(
        row.values if hasattr(row, "values") else row,
        predict_fn,
        num_features=num_features,
    )

    pred_proba = predict_fn(row_batch)
    predicted_class = int(np.argmax(pred_proba)) if pred_proba.ndim == 2 else 0

    actual_value = None
    if y_test is not None:
        actual_value = y_test.iloc[instance_idx] if hasattr(y_test, "iloc") else y_test[instance_idx]

    # 1-row DataFrame for UI display
    if hasattr(X_test, "iloc"):
        X_row_df = X_test.iloc[instance_idx : instance_idx + 1]
    else:
        X_row_df = pd.DataFrame([X_test[instance_idx]], columns=feature_names)

    return explanation, pred_proba, predicted_class, actual_value, X_row_df, class_names


# ---------- Streamlit-facing UI ----------

def display_lime(model):
    st.write("### LIME Explanation for", model["name_of_model"])

    # high-level validation with nice messages
    try:
        # this will raise ValueError with a friendly message if something is missing
        _get_lime_inputs(model)
    except ValueError as e:
        st.warning(str(e))
        return
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return

    X_test = st.session_state["X_test"]

    with st.expander("View LIME Explanation", expanded=True):
        instance_idx = st.select_slider(
            "Select test instance to explain",
            options=list(range(len(X_test))),
            value=min(5, len(X_test) - 1),
            format_func=lambda x: f"Instance {x + 1}",
        )
        try:
            (
                explanation,
                pred_proba,
                predicted_class,
                actual_value,
                X_row_df,
                class_names,
            ) = compute_lime_explanation(model, instance_idx, num_features=10)
        except Exception as e:
            st.error(f"LIME explanation failed: {str(e)}")
            return

        st.write("### Selected Instance Details")
        st.dataframe(X_row_df)

        if actual_value is not None:
            st.write(f"**Actual Value:** {actual_value}")

        with card_container(key="lime_explanation"):
            st.write("")
            st.write(f"## LIME Explanation for Instance {instance_idx + 1}")
            fig = explanation.as_pyplot_figure()
            st.pyplot(fig)

            if pred_proba is not None and pred_proba.ndim == 2 and pred_proba.shape[1] > 0:
                st.write(
                    f"- Predicted class: **{class_names[predicted_class]}** "
                    f"({pred_proba[0][predicted_class]:.1%})"
                )

            if actual_value is not None and 0 <= int(actual_value) < len(class_names):
                st.write(f"**Actual Label:** {class_names[int(actual_value)]}")

            if pred_proba is not None and pred_proba.ndim == 2 and pred_proba.shape[1] > 1:
                st.write("### Prediction Probabilities")
                proba_df = pd.DataFrame(pred_proba, columns=class_names)
                st.dataframe(
                    proba_df.style.format("{:.2%}").highlight_max(axis=1, color="lightgreen")
                )
                st.markdown(
                    """
                    **How to interpret (LIME):**
                    - Green bars support the model’s prediction; red bars go against it.  
                    - The longer the bar, the more impact that feature had on this prediction.  
                    - Usually, the top few features explain most of what drove the decision.  
                    - Comparing what the model emphasizes with what *you* expect can help reveal bias or odd patterns in behavior.
                    """
                )