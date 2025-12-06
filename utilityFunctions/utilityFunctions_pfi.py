import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from local_components import card_container

# ---------- core helpers (logic) ----------

def _resolve_model(model):
    """Accept either the session 'model dict' or a raw estimator."""
    return model["model"] if isinstance(model, dict) and "model" in model else model

def _get_pfi_inputs():
    """
    Pull X_test, y_test from session state.
    Raises ValueError with a friendly message if missing.
    """
    if "X_test" not in st.session_state or "y_test" not in st.session_state:
        raise ValueError("Test data not found. Please train the model first.")

    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    return X_test, y_test

def compute_pfi_importance(model, scoring: str = "accuracy", n_repeats: int = 20):
    """
    Core PFI computation.

    Returns:
        importance_df: DataFrame with Feature, Mean Importance, Std Importance
    """
    X_test, y_test = _get_pfi_inputs()
    est = _resolve_model(model)

    result = permutation_importance(
        est,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Mean Importance": result.importances_mean,
            "Std Importance": result.importances_std,
        }
    ).sort_values(by="Mean Importance", ascending=False)

    return importance_df

# ---------- Streamlit-facing UI ----------

def display_pfi(model):
    st.write("### Permutation Feature Importance (PFI) Analysis")

    # high-level validation
    try:
        _get_pfi_inputs()
    except ValueError as e:
        st.warning(str(e))
        return
    except Exception as e:
        st.error(f"Failed to load test data: {str(e)}")
        return

    with st.expander("View Permutation Feature Importances", expanded=True):
        with card_container(key="pfi_plot_card"):
            st.write("#### Permutation Feature Importances")

            scoring_metric = st.selectbox(
                "Scoring metric",
                options=["accuracy", "f1", "roc_auc"],
                index=0,
                key="pfi_scoring",
            )
            n_repeats = st.slider(
                "Number of shuffle repetitions",
                min_value=5,
                max_value=50,
                value=20,
                key="pfi_n_repeats",
            )
            with st.spinner("Calculating permutation importances..."):
                try:
                    importance_df = compute_pfi_importance(
                        model,
                        scoring=scoring_metric,
                        n_repeats=n_repeats,
                    )
                    num_features = st.slider(
                        "Number of features to display",
                        min_value=1,
                        max_value=len(importance_df),
                        value=min(15, len(importance_df)),
                        key="pfi_feature_slider",
                    )
                    with card_container(key="inside_pfi_plot"):
                        st.write("")
                        plot_df = importance_df.head(num_features)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(
                            data=plot_df,
                            x="Mean Importance",
                            y="Feature",
                            ax=ax,
                        )
                        ax.errorbar(
                            plot_df["Mean Importance"],
                            plot_df["Feature"],
                            xerr=plot_df["Std Importance"],
                            fmt="none",
                            ecolor="gray",
                            capsize=3,
                        )
                        ax.set_title("Top Permutation Feature Importances")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.clf()

                        st.markdown(
                            """
                            **How to interpret (PFI):**
                            - Each bar shows how much model performance drops when that feature is randomly shuffled.  
                            - Larger drops mean the feature is more important to the model’s predictions.  
                            - Small or near-zero bars mean the model doesn’t rely heavily on that feature.  
                            - The gray error bars show variation across multiple shuffles, which helps judge stability.
                            """
                        )
                except Exception as e:
                    st.error(f"Failed to calculate permutation importance: {str(e)}")