import streamlit as st

st.set_page_config(
    page_title="Biosignal Data Analysis App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import only the necessary modules
from dashboard import gsr_ppg_app
from chatBot import run_chatbot

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
main_selection = st.sidebar.radio("Go to", ["Home", "BioSignal Analysis", "Offline Assistant"])

# --- Navigation logic ---
if main_selection == "Home":
    st.title("Welcome to the Cognitive Load & Biosignal App")
    st.write("""
    This app allows you to analyze GSR and PPG biosignal data for stress and physiological trends.
    You can also use the multimodal AI assistant that combines image and emotional data to offer real-time suggestions.
    Use the menu on the left to get started.
    """)

elif main_selection == "BioSignal Analysis":
    sub_selection = st.sidebar.selectbox("Select Analysis Type", [
        "GSR/PPG Analysis"
    ])

    if sub_selection == "GSR/PPG Analysis":
        gsr_ppg_app()

elif main_selection == "Offline Assistant":
    run_chatbot()
