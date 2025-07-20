# dashboard.py
# integrated_gsr_ppg_app with ML-based emotional inference

import os
import time
import csv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import streamlit as st
import logging
import joblib
from serial import Serial, SerialException
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from serial.tools import list_ports
from queue import Queue

# Suppress matplotlib warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Sampling rates
gsr_sampling_rate = 1000
ppg_sampling_rate = 1000

DATA_STORAGE_DIR = 'uploaded_data_gsr_ppg'
EMOTIONAL_STATE_FILE = 'C:\\Users\\dtfygu876\\prompt_codes\\csvChunking\\Chatbot_for_Biosensor\\emotional_state.txt'

# Ensure data storage path exists
os.makedirs(DATA_STORAGE_DIR, exist_ok=True)
data_queue = Queue()
sdnn_values = []
sdnn_timestamps = []
gsr_values = []
gsr_timestamps = []

# Load ML emotion model
@st.cache_resource
def load_emotion_model():
    model_path = os.path.join(os.path.dirname(__file__), "emotion_rf_model.pkl")
    return joblib.load(model_path)

emotion_model = load_emotion_model()

# Save emotional state with timestamp
def save_emotional_state(emotional_state, sdnn=None, gsr_mean=None, output_file=EMOTIONAL_STATE_FILE):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, 'a', encoding='utf-8') as file:
        file.write(f"=== Emotional Analysis Entry ===\n")
        file.write(f"Timestamp: {timestamp_str}\n")
        if sdnn is not None:
            file.write(f"HRV_SDNN: {sdnn:.2f} ms\n")
        if gsr_mean is not None:
            file.write(f"GSR_Mean: {gsr_mean:.4f} μS\n")
        file.write(f"State Inference: {emotional_state}\n\n")
    st.success(f"Emotional state and metrics saved in `{output_file}`.")

# ML prediction
def infer_emotional_state_ml(sdnn, gsr_mean):
    if sdnn is None or gsr_mean is None:
        return "Unknown"
    input_data = pd.DataFrame([[sdnn, gsr_mean]], columns=["sdnn", "gsr_mean"])
    return emotion_model.predict(input_data)[0]

# Plot SDNN trend
def show_sdnn_fluctuations():
    st.subheader("📈 HRV Fluctuation Dashboard (SDNN)")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(sdnn_timestamps, sdnn_values, marker='o', linestyle='-', color='blue')
    ax.set_ylabel("SDNN (ms)")
    ax.set_xlabel("Time")
    ax.set_title("SDNN (HRV) Over Time")
    ax.grid(True)
    st.pyplot(fig)

# Plot GSR trend
def show_gsr_fluctuations():
    st.subheader("🧪 GSR Conductance Fluctuation Dashboard")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(gsr_timestamps, gsr_values, marker='o', linestyle='-', color='green')
    ax.set_ylabel("GSR Conductance (normalized)")
    ax.set_xlabel("Time")
    ax.set_title("GSR Fluctuation Over Time")
    ax.grid(True)
    st.pyplot(fig)

# Core analysis function
def analyze_gsr_ppg_data(gsr_ppg_data):
    scaler = StandardScaler()
    gsr_ppg_data['GSR_Skin_Conductance_CAL'] = scaler.fit_transform(gsr_ppg_data[['GSR_Skin_Conductance_CAL']])
    gsr_ppg_data['PPG_A13_CAL'] = scaler.fit_transform(gsr_ppg_data[['PPG_A13_CAL']])

    st.subheader('📄 Data Sample')
    st.write(gsr_ppg_data.head())

    # HRV/SDNN
    st.header('PPG HRV Analysis')
    ppg_signal = gsr_ppg_data['PPG_A13_CAL'].values
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=ppg_sampling_rate)
    ppg_peaks = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=ppg_sampling_rate)
    ppg_peak_indices = ppg_peaks['PPG_Peaks']

    if len(ppg_peak_indices) > 1:
        intervals = np.diff(ppg_peak_indices) * (1000 / ppg_sampling_rate)
        sdnn = np.std(intervals)
        st.write(f"🔹 SDNN: {sdnn:.2f} ms")
        sdnn_values.append(sdnn)
        sdnn_timestamps.append(datetime.datetime.now().strftime('%H:%M:%S'))
        show_sdnn_fluctuations()
    else:
        st.warning("Not enough PPG peaks to compute SDNN.")
        return

    # GSR
    st.header("GSR Analysis")
    gsr_current = np.mean(gsr_ppg_data['GSR_Skin_Conductance_CAL'].values)
    gsr_values.append(gsr_current)
    gsr_timestamps.append(datetime.datetime.now().strftime('%H:%M:%S'))
    show_gsr_fluctuations()

    # Emotion Inference
    st.header("🧠 Predicted Emotional State")
    emotion = infer_emotional_state_ml(sdnn, gsr_current)
    st.markdown(f"### 👉 **{emotion.upper()}**")
    save_emotional_state(emotion, sdnn, gsr_current)

# Main dashboard app
def gsr_ppg_app():
    st.title("GSR and PPG Real-Time Emotion Dashboard")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    def process_and_analyze(df):
        col_map = {'Timestamp': 'Timestamp_Unix_CAL', 'GSR_Value': 'GSR_Skin_Conductance_CAL', 'PPG_Value': 'PPG_A13_CAL'}
        for orig, new in col_map.items():
            if orig in df.columns:
                df.rename(columns={orig: new}, inplace=True)

        required_cols = ['Timestamp_Unix_CAL', 'GSR_Skin_Conductance_CAL', 'PPG_A13_CAL']
        if not all(col in df.columns for col in required_cols):
            st.error("Missing required columns in file.")
            return

        df.dropna(inplace=True)
        df['Timestamp_Unix_CAL'] = pd.to_numeric(df['Timestamp_Unix_CAL'], errors='coerce')
        df['Timestamp'] = pd.to_datetime(df['Timestamp_Unix_CAL'], unit='ms', errors='coerce')
        df.set_index('Timestamp', inplace=True)

        filename = f"gsr_ppg_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(os.path.join(DATA_STORAGE_DIR, filename), encoding='utf-8')
        st.write(f"Data saved as `{filename}` in `{DATA_STORAGE_DIR}`.")
        analyze_gsr_ppg_data(df)

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', encoding_errors='replace')
            process_and_analyze(df)
        except Exception as e:
            st.error(f"Error: {e}")

    st.header("Re-analyze Stored Data")
    stored_files = [f for f in os.listdir(DATA_STORAGE_DIR) if f.endswith('.csv')]
    if stored_files:
        selected_file = st.selectbox("Select a stored file", stored_files)
        if st.button("Analyze Selected File"):
            try:
                df = pd.read_csv(os.path.join(DATA_STORAGE_DIR, selected_file), encoding='utf-8', encoding_errors='replace')
                process_and_analyze(df)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("No stored files available.")

if __name__ == "__main__":
    gsr_ppg_app()
