import streamlit as st
import utilityFunctions as progress

st.set_page_config(layout="centered", page_title="Home", page_icon="🏠")
progress.render_sidebar(read_only=True)

st.subheader("Welcome to iML Studio 🎓")

st.write("""
The **Machine Learning Studio (MLSP)** is an interactive web platform that enables **students, researchers, and professionals** in fields such as **healthcare, education, and data science** to build, visualize, and interpret machine learning models without requiring extensive coding experience.  
It provides tools for **data preprocessing**, **exploratory data analysis (EDA)**, **model training**, and **interpretability**, helping users understand how models make predictions and what factors influence those decisions.
""")

st.write("")  # light vertical space

st.markdown("""
### 🔍 Key Features
- **Data Preprocessing:** Upload, clean, and normalize your datasets  
- **Exploratory Data Analysis (EDA):** Explore correlations, distributions, and key patterns  
- **Model Training:** Build and evaluate machine learning models  
- **Interpretability:** Visualize SHAP, LIME, PDP, and PFI explanations  
- **Evaluation:** Review metrics, confusion matrices, and charts  
- **Tuning:** Adjust hyperparameters interactively for better performance
""")

st.write("")

st.markdown("""
### 💡 Quick Start
*(You can also follow progress on the left sidebar)*  
1. Open **Preprocess Data** and load a dataset  
2. Select your **target column**  
3. Explore the data in the **EDA** section  
4. Train a model in one of the **model pages**  
5. Visualize and interpret results
""")

st.write("")

st.markdown("""
### 🧩 Troubleshooting
If you encounter issues or bugs, try the following:
- Refresh the browser  
- Clear cached state or reload the app  
- Make sure a **target** column is selected  
- Verify your dataset has clean headers and consistent data types  

**Contact:** chojon99@uw.edu for any additional help or support.
""")

st.write("")

st.markdown("""
### 📝 [Feedback](https://forms.gle/c4F2mdaszrqJKNzQA)
We value your feedback to help improve this platform.  
Please share your thoughts and suggestions using the form below:  
""")

st.warning("Note: The platform is currently in beta. Some features are still being optimized.")
