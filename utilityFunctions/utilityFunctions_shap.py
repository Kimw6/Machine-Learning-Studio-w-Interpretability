import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from local_components import card_container
import streamlit_shadcn_ui as ui  # ok to keep even if not used much

# ---------- core helpers (logic) ----------

def get_shap_explainer(model, X_train=None):
    """Choose an appropriate SHAP explainer based on model type."""
    model = model["model"] if isinstance(model, dict) and "model" in model else model
    t = str(type(model)).lower()

    if any(keyword in t for keyword in ["tree", "forest", "booster", "gbm"]):
        return shap.TreeExplainer(model)
    elif hasattr(model, "coef_"):
        return shap.LinearExplainer(model, X_train)
    elif "keras" in t or "torch" in t:
        return shap.DeepExplainer(model, X_train[:100])
    else:
        background = shap.sample(X_train, 100) if X_train is not None else None
        return shap.KernelExplainer(
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            background,
        )

def _quantile_labels(
    values: pd.Series,
    q: int,
    scheme: str = "auto",
) -> tuple[list[str], list[str]]:
    """
    Returns:
      cohort_labels_per_row: list[str] aligned to values index
                             (e.g., "Low: 1.23–2.34 (28, 25%)")
      legend_labels:        list[str] one per bin for reference/legend
    """
    bins = pd.qcut(values, q=q, duplicates="drop")
    edges = bins.cat.categories
    counts = pd.value_counts(bins, sort=False)
    total = float(len(values))

    if scheme == "low-med-high" and len(edges) == 3:
        names = ["Low", "Medium", "High"]
    elif scheme == "quartiles" and len(edges) == 4:
        names = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    else:
        names = [f"Bin {i+1}" for i in range(len(edges))]

    legend_labels = []
    mapping = {}
    for i, edge in enumerate(edges):
        left, right = edge.left, edge.right
        n = int(counts.iloc[i])
        pct = (n / total) * 100.0
        pretty = f"{names[i]}: {left:.2f}–{right:.2f} ({n}, {pct:.0f}%)"
        legend_labels.append(pretty)
        mapping[edge] = pretty

    cohort_labels_per_row = pd.Series(bins).map(mapping).tolist()
    return cohort_labels_per_row, legend_labels

def _get_shap_inputs(model):
    """
    Pull X_train, X_test and underlying estimator.
    Raises ValueError with a friendly message if something is missing.
    """
    if "X_train" not in st.session_state or "X_test" not in st.session_state:
        raise ValueError("Please run model training first.")

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]

    est = model["model"] if isinstance(model, dict) and "model" in model else model
    return est, X_train, X_test

def compute_shap_values(model):
    """
    Core SHAP computation.

    Returns:
        shap_values : SHAP Explanation object / array
        max_feats   : max feature count from underlying SHAP values
        X_train, X_test
    """
    est, X_train, X_test = _get_shap_inputs(model)
    explainer = get_shap_explainer(est, X_train)

    shap_values = explainer(X_test)
    base_vals = getattr(shap_values, "values", shap_values)
    max_feats = base_vals.shape[1]

    return shap_values, max_feats, X_train, X_test

# ---------- Streamlit-facing UI ----------

def display_shap(model):
    st.write("### SHAP Explanation")

    # High-level validation
    try:
        # Just to trigger validation; we’ll recompute below
        _get_shap_inputs(model)
    except ValueError as e:
        st.warning(str(e))
        return
    except Exception as e:
        st.error(f"Failed to prepare data for SHAP: {str(e)}")
        return

    try:
        with st.spinner("Calculating SHAP values..."):
            shap_values, _max_feats, X_train, X_test = compute_shap_values(model)

            # handle multiclass shape
            if len(shap_values.shape) == 3:
                class_idx = st.selectbox(
                    "Select class to explain",
                    options=list(range(shap_values.shape[2])),
                    format_func=lambda x: f"Class {x}",
                )
                shap_values = shap_values[..., class_idx]

            max_features = shap_values.shape[1]

            # --- Global bar plot ---
            with st.expander("View global SHAP plots", expanded=False):
                with card_container(key="global_shap_bar_inside"):
                    st.write("")
                    st.write("#### Global bar plot")
                    num_features = st.slider(
                        "Number of features to display",
                        min_value=1,
                        max_value=max_features,
                        value=min(15, max_features),
                        key="shap_feature_slider",
                    )
                    fig, _ = plt.subplots(figsize=(10, 6))
                    shap.plots.bar(shap_values, max_display=num_features, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                    st.markdown(
                        """
                        **How to interpret (global SHAP bar plot):**
                        - Each bar represents how much a feature contributes to the model’s predictions on average.  
                        - Longer bars mean the feature has a stronger influence overall.  
                        - This plot shows importance, not direction — it tells you *how much* a feature matters, not *whether it increases or decreases* predictions.  
                        - Use this to quickly identify which features the model relies on most.
                        """
                    )

            # --- Cohort comparison (quantiles OR SHAP auto-cohort) ---
            with st.expander("View cohort comparison bar plot", expanded=False):
                with card_container(key="cohort_shap_bar_inside"):
                    st.write("")

                    mode = st.radio(
                        "Choose cohorting mode",
                        ["By feature quantiles", "Auto (tree-based, SHAP)"],
                        index=0,
                        key="shap_cohort_mode",
                    )

                    if mode == "By feature quantiles":
                        feat = st.selectbox(
                            "Feature for cohorting",
                            X_test.columns,
                            key="shap_cohort_feat",
                        )
                        q = st.slider(
                            "Number of bins",
                            min_value=2,
                            max_value=5,
                            value=3,
                            key="shap_cohort_q",
                        )
                        label_style = st.selectbox(
                            "Label style",
                            [
                                "Auto bins",
                                "Low/Medium/High (3 bins)",
                                "Quartiles (4 bins)",
                            ],
                            index=0,
                            key="shap_cohort_style",
                        )
                        scheme = (
                            "low-med-high"
                            if label_style.startswith("Low/Medium/High")
                            else "quartiles"
                            if label_style.startswith("Quartiles")
                            else "auto"
                        )

                        try:
                            values = X_test[feat]
                            cohort_labels, legend_labels = _quantile_labels(
                                values, q=q, scheme=scheme
                            )

                            cohort_expl = shap_values.cohorts(cohort_labels)

                            st.markdown("**Cohorts:**")
                            for lbl in legend_labels:
                                st.write(f"- {lbl}")

                            st.subheader("Cohort comparison by feature ranges")
                            fig = plt.figure(figsize=(10, 6))
                            shap.plots.bar(
                                cohort_expl.abs.mean(0),
                                show=False,
                                max_display=min(15, _max_feats),
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            st.markdown(
                                """
                                How to interpret (SHAP cohorts by quantiles):

                                - Each cohort represents a range of values for the selected feature, like low, medium, or high.
                                - The bars show the average absolute SHAP value for each feature within that range.
                                - Comparing cohorts helps you see how feature importance changes across different ranges.
                                - The labels include the range, number of samples, and percentage in each group.
                                - This helps you understand how the model behaves differently across sections of the data.
                                """
                            )
                        except Exception as e:
                            st.info(f"Cohort comparison unavailable: {e}")
                    else:
                        n = st.slider(
                            "Number of cohorts (auto)",
                            min_value=2,
                            max_value=5,
                            value=2,
                            key="shap_auto_cohort_n",
                            help=(
                                "SHAP splits instances into N cohorts to "
                                "maximize separation of SHAP patterns."
                            ),
                        )
                        try:
                            cohort_expl = shap_values.cohorts(
                                n
                            )  # SHAP auto-names cohorts with rule + [count]

                            st.markdown("**Cohorts (auto):**")
                            for name in cohort_expl.cohorts.keys():
                                st.write(f"- {name}")

                            st.subheader("Cohort comparison by SHAP auto-clustering")

                            fig = plt.figure(figsize=(10, 6))
                            shap.plots.bar(
                                cohort_expl.abs.mean(0),
                                show=False,
                                max_display=min(15, _max_feats),
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            st.markdown(
                                """
                                How to interpret (SHAP auto cohorts):

                                - Each cohort groups similar data points based on how the model behaves, not on one specific feature.  
                                - Bars summarize the average absolute SHAP values for each cohort.  
                                - Comparing cohorts helps you see where the model’s logic differs between groups.  
                                - Cohorts are automatically discovered based on patterns in feature influence.  
                                - This view helps reveal differences in behavior or bias across subgroups in your data.
                                """
                            )
                        except Exception as e:
                            st.info(
                                "Auto-cohorting unavailable: "
                                f"{e} — ensure your SHAP version supports "
                                "`Explanation.cohorts(N)`."
                            )

            # --- Beeswarm summary ---
            with st.expander("View beeswarm summary plots", expanded=False):
                with card_container(key="beeswarm_shap_plot_inside"):
                    st.write("")
                    st.write("#### Beeswarm summary plot")
                    num_features = st.slider(
                        "Number of features to display",
                        min_value=1,
                        max_value=max_features,
                        value=min(15, max_features),
                        key="shap_beeswarm_slider",
                    )
                    fig, _ = plt.subplots(figsize=(10, 6))
                    shap.plots.beeswarm(
                        shap_values, max_display=num_features, show=False
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                    st.markdown(
                        """
                        **How to interpret (beeswarm plot):**
                        - Each point represents one instance (a row from your data).  
                        - The x-axis shows how much that feature pushes the prediction higher or lower.  
                        - Red points indicate higher feature values; blue points indicate lower ones.  
                        - If red points appear mostly on the right, higher values increase the prediction.  
                        - If they’re on the left, higher values lower it.  
                        - This helps visualize both direction and strength of influence across many samples.
                        """
                    )

            # --- Local SHAP ---
            with st.expander("View local SHAP plots", expanded=False):
                with card_container(key="local_shap_bar_inside"):
                    st.write("")
                    st.write("#### Local bar plot")
                    num_features = st.slider(
                        "Number of features to display",
                        min_value=1,
                        max_value=max_features,
                        value=min(15, max_features),
                        key="shap_local_slider",
                    )
                    fig, _ = plt.subplots(figsize=(10, 6))
                    shap.plots.bar(
                        shap_values[0],
                        max_display=num_features,
                        show=False,
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                    st.markdown(
                        """
                        **How to interpret (local SHAP):**
                        - This explains how the model made a specific prediction.  
                        - Positive SHAP values push the prediction toward a higher output (for example, predicting “Yes”).  
                        - Negative values push it toward a lower output (for example, predicting “No”).  
                        - The longer the bar, the more that feature influenced this single prediction.  
                        - It’s helpful for checking if the model’s reasoning for one case makes sense.
                        """
                    )
    except Exception as e:
        st.error(f"SHAP calculation failed: {str(e)}")
