import streamlit as st
from ydata_profiling import ProfileReport
import pandas as pd
import hashlib
import utilityFunctions as progress
from local_components import card_container
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Exploratory Data Analysis", page_icon="📊")
progress.render_sidebar(read_only=True)
st.title("Exploratory Data Analysis (EDA)")

def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

@st.cache_data(ttl=3600, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df})
def generate_eda_html(df: pd.DataFrame) -> str:
    profile = ProfileReport(
        df,
        title="Exploratory Data Analysis",
        explorative=True,
        minimal=True,
        html={"navbar_show": False, "style": {"full_width": True}},
        correlations={
            "auto": {"calculate": True},
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True},
        },
        missing_diagrams={"bar": True, "matrix": True, "heatmap": True},
        interactions={"continuous": False},
    )
    return profile.to_html()

def display_cached_report(df: pd.DataFrame):
    html = generate_eda_html(df)
    
    if not st.session_state.get("_qs_eda_marked"):
        progress.set_done("qs_eda", True)
        st.session_state["_qs_eda_marked"] = True
        st.rerun()

    with st.expander("View EDA Report", expanded=True):
        with card_container(key="EDA Report"):
            st.write("")
            components.html(html, height=800, scrolling=True)

    sig = st.session_state.get('_data_sig', 'no_data')
    st.download_button(
        label="💾 Download Full EDA Report (HTML)",
        data=html,
        file_name="eda_report.html",
        mime="text/html",
        key=f"eda-download-{sig}",
        help="Download complete exploratory data analysis report"
    )

# Main app logic
if st.session_state.get('df') is not None and st.session_state.get('target') is not None:
    display_cached_report(st.session_state['df'])
else:
    st.error("Oops, looks like your data is unavailable.\n\nPlease try uploading your data again.", icon="⚠️")
    if st.button("Re-upload Data", key="page_link_btn"):
        st.switch_page("pages/1Preprocess_Data.py")
