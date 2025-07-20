import streamlit as st
import os

# Set page config first
st.set_page_config(
    page_title="Biosignal Data Analysis App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Diagnostic output to help identify issues
st.sidebar.write(f"Current directory: {os.getcwd()}")
st.sidebar.write(f"Files in directory: {', '.join(os.listdir())}")

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
    # Lazy import for dashboard
    try:
        from dashboard import gsr_ppg_app
    except ImportError as e:
        st.error(f"Failed to import dashboard module: {e}")
        st.stop()
    
    sub_selection = st.sidebar.selectbox("Select Analysis Type", [
        "GSR/PPG Analysis"
    ])

    if sub_selection == "GSR/PPG Analysis":
        gsr_ppg_app()

elif main_selection == "Offline Assistant":
    # Lazy import for chatbot with error handling
    try:
        from chatBot import run_chatbot
        run_chatbot()
    except ImportError as e:
        st.error(f"Failed to import chatbot module: {e}")
        st.error("Please ensure chatBot.py exists in the current directory")
        st.code("Current files: " + ", ".join(os.listdir()))
    except Exception as e:
        st.error(f"Error in chatbot: {e}")
