import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from local_components import card_container
import streamlit_shadcn_ui as ui


# ---------- core helpers (logic) ----------

def _get_test_data(model):
    """
    Prefer per-model X_test/y_test, fall back to shared session.
    Raises ValueError if no test data is available.
    """
    X_test = model.get("X_test") if isinstance(model, dict) else None
    y_test = model.get("y_test") if isinstance(model, dict) else None

    if X_test is None or y_test is None:
        X_test = st.session_state.get("X_test")
        y_test = st.session_state.get("y_test")

    if X_test is None or y_test is None:
        raise ValueError("Test data not found. Please train the model first.")

    return X_test, y_test

def build_classification_report_df(metrics_report: dict) -> pd.DataFrame:
    """
    Turn sklearn's classification_report(output_dict=True)
    into a formatted DataFrame for display.
    """
    report_df = pd.DataFrame(metrics_report).transpose().reset_index()
    report_df.rename(columns={"index": "Class"}, inplace=True)

    report_df["precision"] = report_df["precision"].apply(
        lambda x: "{:.2f}".format(x) if not pd.isna(x) else ""
    )
    report_df["recall"] = report_df["recall"].apply(
        lambda x: "{:.2f}".format(x) if not pd.isna(x) else ""
    )
    report_df["f1-score"] = report_df["f1-score"].apply(
        lambda x: "{:.2f}".format(x) if not pd.isna(x) else ""
    )
    report_df["support"] = report_df["support"].apply(
        lambda x: "{:.0f}".format(x) if not pd.isna(x) else ""
    )

    return report_df

def supports_probability_curves(estimator) -> bool:
    """
    Check if estimator can be used for ROC / PR curves.
    """
    return hasattr(estimator, "predict_proba") or hasattr(estimator, "decision_function")

# ---------- Streamlit-facing UI ----------

def display_results(model):
    try:
        X_test, y_test = _get_test_data(model)
    except ValueError as e:
        st.warning(str(e))
        return
    except Exception as e:
        st.error(f"Failed to load test data: {str(e)}")
        return

    st.write("#### Model Performance Metrics")
    cols = st.columns(4)

    with cols[0]:
        ui.card(
            title="Accuracy",
            content="{:.2f}%".format(model["accuracy"] * 100),
            description="Overall model accuracy",
            key="accuracy_card",
        ).render()
    with cols[1]:
        ui.card(
            title="Precision (Avg)",
            content="{:.2f}".format(model["metrics_report"]["macro avg"]["precision"]),
            description="Precision across classes",
            key="precision_card",
        ).render()
    with cols[2]:
        ui.card(
            title="Recall (Avg)",
            content="{:.2f}".format(model["metrics_report"]["macro avg"]["recall"]),
            description="Recall across classes",
            key="recall_card",
        ).render()
    with cols[3]:
        ui.card(
            title="F1-Score (Avg)",
            content="{:.2f}".format(model["metrics_report"]["macro avg"]["f1-score"]),
            description="F1-score across classes",
            key="f1_card",
        ).render()

    # full classification report
    with st.expander("View Full Classification Report", expanded=True):
        st.write("### Classification Report for", model["name_of_model"])
        report_df = build_classification_report_df(model["metrics_report"])

        with card_container(key="classification_report"):
            st.write("")
            ui.table(data=report_df, maxHeight=300)

    # confusion matrix
    with st.expander("View Confusion Matrix"):
        st.write("### Confusion Matrix for", model["name_of_model"])
        with card_container(key="confusion_matrix"):
            st.write("")
            fig, ax = plt.subplots()
            sns.heatmap(model["confusion_matrix"], annot=True, fmt="d", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()

    est = model["model"]

    # curves if supported
    if supports_probability_curves(est):
        with st.expander("View ROC Curve"):
            st.write("### ROC Curve for", model["name_of_model"])
            with card_container(key="roc_curve"):
                st.write("")
                fig, ax = plt.subplots()
                RocCurveDisplay.from_estimator(est, X_test, y_test, ax=ax)
                ax.set_title("ROC Curve")
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()

        with st.expander("View Precision-Recall Curve"):
            st.write("### Precision-Recall Curve for", model["name_of_model"])
            with card_container(key="precision_recall_curve"):
                st.write("")
                fig, ax = plt.subplots()
                PrecisionRecallDisplay.from_estimator(est, X_test, y_test, ax=ax)
                ax.set_title("Precision-Recall Curve")
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
    else:
        st.warning("ROC and Precision-Recall curves are not supported for this model.")
