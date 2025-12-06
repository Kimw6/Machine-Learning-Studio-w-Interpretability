from typing import Optional
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # still imported if you ever want styling tweaks
from sklearn.inspection import PartialDependenceDisplay
from local_components import card_container

# ---------- core helpers (logic) ----------

def _get_pdp_inputs(model):
    """
    Pull df, target, X and estimator from session/model.
    Raises ValueError with a friendly message if something is missing.
    """
    if "df" not in st.session_state or st.session_state["df"] is None:
        raise ValueError("No dataset found. Please upload data first.")
    if "target" not in st.session_state:
        raise ValueError("Target not set. Please select a target variable.")
    if "X_train" not in st.session_state:
        raise ValueError("Please train the model first.")

    df = st.session_state["df"]
    target = st.session_state["target"]
    X = df.drop(target, axis=1)

    est = model["model"] if isinstance(model, dict) and "model" in model else model
    return df, target, X, est

def compute_pdp_figure(
    model,
    primary_feature: str,
    secondary_feature: Optional[str],
    kind: str = "average",
    n_points: int = 10,
):
    """
    Core PDP computation. Returns:

        fig          : matplotlib Figure
        target       : target column name (for titles)
        effective_kind : kind actually used ("average"/"individual"/"both")
        switched     : True if we had to switch kind because of 2D PDP
    """
    df, target, X, est = _get_pdp_inputs(model)

    effective_kind = kind
    switched = False

    # sklearn only supports "average" for 2D PDP
    if secondary_feature is not None and effective_kind != "average":
        effective_kind = "average"
        switched = True

    fig, ax = plt.subplots(figsize=(10, 6))

    if secondary_feature is None:
        PartialDependenceDisplay.from_estimator(
            est,
            X,
            [primary_feature],
            kind=effective_kind,
            grid_resolution=n_points,
            ax=ax,
            random_state=42,
        )
        ax.set_title(f"Partial Dependence of {target} on {primary_feature}")
    else:
        PartialDependenceDisplay.from_estimator(
            est,
            X,
            [(primary_feature, secondary_feature)],
            kind=effective_kind,
            grid_resolution=n_points,
            ax=ax,
            random_state=42,
        )
        ax.set_title(
            f"Partial Dependence of {target}\n"
            f"on {primary_feature} and {secondary_feature}"
        )

    plt.tight_layout()
    return fig, target, effective_kind, switched

# ---------- Streamlit-facing UI ----------

def display_pdp(model):
    st.write("### Partial Dependence Plot (PDP) Analysis")

    # high-level validation
    try:
        df, target, X, est = _get_pdp_inputs(model)
    except ValueError as e:
        st.warning(str(e))
        return
    except Exception as e:
        st.error(f"Failed to prepare data for PDP: {str(e)}")
        return

    with st.expander("Feature Selection", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            selected_feature = st.selectbox(
                "Select primary feature for PDP",
                X.columns,
                index=0,
                key="pdp_feature_select",
            )
        with col2:
            second_feature = st.selectbox(
                "Select secondary feature (optional, for 2D PDP)",
                [None] + list(X.columns),
                index=0,
                key="pdp_second_feature",
            )

    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            n_points = st.slider(
                "Number of grid points",
                min_value=5,
                max_value=50,
                value=10,
            )
        with col2:
            kind = st.radio(
                "Plot type",
                ["average", "individual", "both"],
                index=0,
            )

    with st.spinner("Generating Partial Dependence Plot..."):
        try:
            fig, target, effective_kind, switched = compute_pdp_figure(
                model=model,
                primary_feature=selected_feature,
                secondary_feature=second_feature,
                kind=kind,
                n_points=n_points,
            )

            if second_feature is not None and switched:
                st.info("2D PDP only supports 'average'. Automatically switched.")

            with card_container(key="pdp_plot"):
                st.write("")
                st.pyplot(fig)
                plt.close(fig)

                st.markdown(
                    """
                    **How to interpret (PDP):**
                    - The y-axis shows how the model’s average prediction changes as the selected feature varies.  
                    - An upward trend means higher feature values increase the predicted outcome.  
                    - A flat or near-flat line means that feature doesn’t affect the prediction much.  
                    - In two-feature plots, look for curved or diagonal patterns — these show how two features interact with each other.
                    """
                )

        except Exception as e:
            st.error(f"Failed to generate PDP: {str(e)}")