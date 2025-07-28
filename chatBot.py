import streamlit as st
import os
import time
import threading
import queue
import json
from datetime import datetime
from PIL import Image
import chromadb
from image_processor import analyze_image_with_gradio
from deepMind import DeepMindAgent
import pandas as pd
import logging
import plotly.express as px

# Configure logging
logging.basicConfig(filename='app_debug.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StreamlitApp")

# === Constants and file paths ===
DATA_STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(DATA_STORAGE_DIR, "chroma_data")
DASHBOARD_RESULTS_PATH = r"C:\Users\dtfygu876\prompt_codes\csvChunking\Chatbot_for_Biosensor"
EMOTIONAL_STATE_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "emotional_state.txt")
IMAGE_ANALYSIS_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "imageAnalysis.txt")
CAPTURED_IMAGES_DIR = os.path.join(DASHBOARD_RESULTS_PATH, "captured_images")
FEEDBACK_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "teacher_feedback.csv")
CHART_DATA_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "chart_data.json")

os.makedirs(DASHBOARD_RESULTS_PATH, exist_ok=True)
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)


# === Clients ===
@st.cache_resource(show_spinner=False)
def get_chroma_client():
    logger.info("Creating ChromaDB client")
    return chromadb.PersistentClient(path=CHROMA_PATH)


@st.cache_resource(show_spinner=False)
def get_deepmind_agent():
    logger.info("Creating DeepMind agent")
    return DeepMindAgent()


if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = get_chroma_client()

collection = st.session_state.chroma_client.get_or_create_collection(name="chat_history")
deep_agent = get_deepmind_agent()


# === Utility Functions ===
def save_chat_to_chroma(user_message: str, bot_response: str):
    base_id = f"chat_{datetime.utcnow().isoformat()}"
    metadata = {"timestamp": datetime.utcnow().isoformat(), "source": "avicenna-chatbot"}
    collection.add(
        documents=[user_message, bot_response],
        metadatas=[metadata, metadata],
        ids=[base_id + "_user", base_id + "_bot"],
    )
    logger.info(f"Saved chat to Chroma: {user_message[:50]}...")


def get_emotional_state():
    try:
        if os.path.exists(EMOTIONAL_STATE_FILE):
            with open(EMOTIONAL_STATE_FILE, "r") as f:
                lines = f.readlines()
                if not lines:
                    return "No data"
                latest_block = []
                for line in reversed(lines):
                    if line.strip() == "=== Emotional Analysis Entry ===":
                        break
                    latest_block.insert(0, line.strip())
                return "<br>".join(latest_block)
        return "Unknown"
    except Exception as e:
        logger.error(f"Emotional state error: {str(e)}")
        return f"Error reading emotional state: {e}"


def analyze_and_save_image(image_bytes: bytes):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename = f"captured_{timestamp}.jpg"
    img_path = os.path.join(CAPTURED_IMAGES_DIR, img_filename)

    try:
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        with st.spinner("🧠 Analyzing image with VLM..."):
            analysis_result = analyze_image_with_gradio(img_path)

            if isinstance(analysis_result, dict) and "error" in analysis_result:
                raise RuntimeError(analysis_result["error"])
            elif isinstance(analysis_result, str) and "error" in analysis_result.lower():
                raise RuntimeError(analysis_result)

            # Save structured results
            log_entry = f"{timestamp}|{img_path}|{json.dumps(analysis_result)}\n"
            with open(IMAGE_ANALYSIS_FILE, "a", encoding="utf-8") as f:
                f.write(log_entry)

        return analysis_result, img_path, None
    except Exception as e:
        error_msg = f"Image processing error: {str(e)}"
        logger.error(error_msg)
        with open(IMAGE_ANALYSIS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}|{img_path}|{error_msg}\n")
        return None, None, error_msg


def display_vlm_analysis(analysis_result):
    """Display VLM analysis in a structured format"""
    if not analysis_result or ("error" in analysis_result if isinstance(analysis_result, dict) else False):
        if isinstance(analysis_result, dict):
            error = analysis_result.get("error", "Unknown error")
        else:
            error = str(analysis_result)
        st.error(f"Analysis error: {error}")
        return

    # Handle both structured dict and legacy string output
    if isinstance(analysis_result, str):
        # Legacy format - display as is
        st.info(analysis_result)
        return

    with st.expander("🔍 VLM Analysis Details", expanded=True):
        cols = st.columns(2)

        with cols[0]:
            st.subheader("🧍 Student Behavior")
            behavior = analysis_result.get("behavior", "No behavior analysis")
            st.info(behavior)

            # Behavior recommendations
            if "leaning back" in behavior.lower():
                st.warning("⚠️ Student posture suggests disengagement - try active learning techniques")
            if "looking away" in behavior.lower():
                st.warning("⚠️ Student attention wandering - try proximity or questioning")
            if "fidgeting" in behavior.lower():
                st.warning("⚠️ Student appears restless - consider movement break")

            st.subheader("💡 Environment")
            environment = analysis_result.get("environment", "No environment analysis")
            st.info(environment)

            # Environment recommendations
            if "dark" in environment.lower():
                st.warning("⚠️ Low lighting may cause eye strain - adjust lighting")
            if "clutter" in environment.lower():
                st.warning("⚠️ Cluttered environment may reduce focus - suggest cleanup")

        with cols[1]:
            st.subheader("🚫 Distractions")
            distractions = analysis_result.get("distractions", [])
            if distractions:
                st.warning(f"⚠️ {len(distractions)} distractions detected:")
                for i, item in enumerate(distractions, 1):
                    st.markdown(f"{i}. {item}")
            else:
                st.success("✅ No distractions detected")

            st.subheader("😔 Emotional Cues")
            emotional_cues = analysis_result.get("emotional_cues", "No emotional cues detected")
            st.info(emotional_cues)

            # Engagement score if available
            if "engagement_score" in analysis_result:
                engagement = analysis_result["engagement_score"]
                st.metric("Engagement Score", f"{engagement}/100")
                st.progress(engagement / 100)

                if engagement < 40:
                    st.error("🔴 Low engagement - intervention needed")
                elif engagement < 70:
                    st.warning("🟡 Medium engagement - monitor closely")
                else:
                    st.success("🟢 High engagement - good focus")


def show_cognitive_dashboard():
    st.subheader("📊 Cognitive Load Dashboard")

    try:
        if not os.path.exists(CHART_DATA_FILE):
            st.warning("No chart data available yet")
            return

        with open(CHART_DATA_FILE, "r") as f:
            chart_data = json.load(f)

        # Extract data
        load_history = chart_data.get("cognitive_load_history", [])

        if not load_history:
            st.info("Dashboard data is being collected. Check back soon.")
            return

        # Create DataFrame
        df = pd.DataFrame(load_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Map load levels to numeric values
        level_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df['load_value'] = df['load_level'].map(level_map).fillna(1.5)

        # Create figure
        fig = px.line(
            df,
            x='timestamp',
            y='load_value',
            markers=True,
            hover_data=['load_level', 'explanation'],
            labels={'load_value': 'Cognitive Load Level', 'timestamp': 'Time'},
            title='Cognitive Load Over Time'
        )

        # Customize y-axis
        fig.update_yaxes(
            tickvals=list(level_map.values()),
            ticktext=list(level_map.keys()),
            range=[0.5, 3.5]
        )

        # Add custom hover template
        fig.update_traces(
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>Level: %{customdata[0]}<br>%{customdata[1]}"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
        logger.error(f"Dashboard error: {str(e)}")


def teacher_tools():
    st.subheader("🧑‍🏫 Teacher Tools")

    # Cognitive Load Assessment
    with st.expander("📝 Cognitive Load Assessment", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            state = st.radio("Cognitive Load State",
                             ["Overloaded", "Medium", "Focus"],
                             index=1)
        with col2:
            level = st.slider("Cognitive Load Level", 0, 100, 50)

        if st.button("💾 Save Assessment"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"""
=== Emotional Analysis Entry ===
Timestamp: {timestamp}
State Inference: {state} - Level {level}
"""
            with open(EMOTIONAL_STATE_FILE, "a", encoding="utf-8") as f:
                f.write(entry)
            st.success("Assessment saved successfully!")

    # Teaching Advisor
    with st.expander("💡 Teaching Advisor", expanded=True):
        task_options = ["Lecture", "Quiz", "Group Discussion", "Self-study", "Practical Work"]
        current_task = st.selectbox("Select current task type:", task_options, key="task_select")
        st.session_state.current_task = current_task

        if st.button("🛠️ Get Teaching Strategies", key="teacher_advice_btn"):
            if 'current_task' in st.session_state:
                with st.spinner("Generating teaching strategies..."):
                    advice = deep_agent.get_teacher_advice(st.session_state.current_task)
                    st.session_state.teacher_advice = advice
            else:
                st.warning("Please select a task type first")

        if st.session_state.get("teacher_advice"):
            st.subheader("Teaching Recommendations")
            st.info(st.session_state.teacher_advice)

    # Cognitive Load Advisor
    with st.expander("🧠 Cognitive Load Advisor", expanded=True):
        st.info("Get personalized suggestions based on current cognitive load")
        if st.button("📊 Generate Advisor Recommendations"):
            with st.spinner("Analyzing cognitive load..."):
                # Get cognitive load analysis
                analysis = deep_agent.get_cognitive_load_analysis()

                if "error" in analysis:
                    st.error(f"Analysis failed: {analysis['error']}")
                else:
                    # Get advisor recommendations
                    advisor_response = deep_agent.get_cognitive_load_advisor(analysis)
                    st.session_state.advisor_response = advisor_response

        if st.session_state.get("advisor_response"):
            st.subheader("Personalized Recommendations")
            st.info(st.session_state.advisor_response)


def teacher_feedback_ui():
    st.subheader("🧑‍🏫 Teacher Feedback")
    st.markdown("**Was this app useful to you?**")
    col1, col2 = st.columns(2)
    with col1:
        useful = st.button("👍 Yes", key="feedback_yes")
    with col2:
        not_useful = st.button("👎 No", key="feedback_no")

    feedback_flag = useful or not_useful
    feedback_value = useful and not not_useful
    suggestion = st.text_area("Any suggestion or comment?", key="suggestion_input")

    if feedback_flag:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feedback_data = pd.DataFrame([{
            "timestamp": timestamp,
            "useful": "Yes" if feedback_value else "No",
            "suggestion": suggestion.strip() or "None"
        }])

        if os.path.exists(FEEDBACK_FILE):
            df_existing = pd.read_csv(FEEDBACK_FILE)
            df_combined = pd.concat([df_existing, feedback_data], ignore_index=True)
        else:
            df_combined = feedback_data

        df_combined.to_csv(FEEDBACK_FILE, index=False)
        st.success("✅ Thank you for your feedback!")
        logger.info(f"Feedback received: useful={feedback_value}")

    if os.path.exists(FEEDBACK_FILE):
        st.markdown("---")
        st.markdown("📋 **Recent Feedback**")
        try:
            df = pd.read_csv(FEEDBACK_FILE)
            st.dataframe(df.tail(5))
        except Exception as e:
            st.error(f"Error reading feedback: {e}")


# === Main Chatbot UI ===
def run_chatbot():
    st.title("🧠 Avicenna - Cognitive Load & Classroom Insight System")

    # Initialize session state keys
    init_keys = [
        "last_analysis", "last_image_path", "last_analysis_time",
        "deepmind_response", "deepmind_error", "last_captured_image",
        "chat_history", "deepmind_processing", "processing_thread",
        "result_queue", "current_user_input", "processing_start_time",
        "current_task", "teacher_advice", "advisor_response"
    ]

    for key in init_keys:
        if key not in st.session_state:
            st.session_state[key] = None if key != "chat_history" else []

    # Main layout columns
    col1, col2 = st.columns([3, 2])

    with col1:
        # Biometric Summary
        emotional_state_text = get_emotional_state()
        with st.expander("📊 Biometric Data Summary", expanded=True):
            st.markdown(f"- **Latest Emotional State:**<br>{emotional_state_text}", unsafe_allow_html=True)

        # Image Capture and Analysis
        st.subheader("📸 Classroom Image Analysis")
        # Changed key here to fix duplicate key issue:
        img_file = st.camera_input("Capture classroom image for VLM analysis", key="camera_input_1")

        if img_file is not None:
            if (st.session_state.last_captured_image != img_file.getvalue() or
                    st.session_state.last_image_path is None):

                st.session_state.last_captured_image = img_file.getvalue()
                image_bytes = img_file.getvalue()
                analysis_result, img_path, error = analyze_and_save_image(image_bytes)
                if error:
                    st.error(f"Image processing failed: {error}")
                else:
                    st.session_state.last_analysis = analysis_result
                    st.session_state.last_image_path = img_path
                    st.session_state.last_analysis_time = datetime.now()
                    st.success("Image captured and analyzed successfully!")

        # Display image and analysis
        if st.session_state.last_analysis:
            try:
                img = Image.open(st.session_state.last_image_path)
                st.image(img, caption=f"📸 {st.session_state.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S')}",
                         use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display image: {str(e)}")

            # Display VLM analysis
            display_vlm_analysis(st.session_state.last_analysis)
        else:
            st.info("No image analysis available. Capture an image to begin.")

        # Cognitive Load Dashboard
        show_cognitive_dashboard()

    with col2:
        # Teacher Tools Section
        teacher_tools()

        # Student Interaction
        st.subheader("💬 Student Interaction")
        user_input = st.text_input("Enter your message:", key="user_input")
        st.session_state.current_user_input = user_input

        # Generate Cognitive Support Plan
        if st.button("🧠 Generate Cognitive Support Plan", key="deepmind_button"):
            if not user_input:
                st.warning("Please enter your message.")
            else:
                logger.info("Starting DeepMind solution generation")
                # Initialize processing state
                st.session_state.deepmind_processing = True
                st.session_state.deepmind_response = None
                st.session_state.deepmind_error = None
                st.session_state.result_queue = queue.Queue()

                # Create and start processing thread
                def run_deepmind():
                    try:
                        result = deep_agent.get_student_report(user_input)
                        st.session_state.result_queue.put(result)
                    except Exception as e:
                        st.session_state.result_queue.put(f"⚠️ Processing Error: {str(e)}")

                st.session_state.processing_thread = threading.Thread(
                    target=run_deepmind,
                    daemon=True
                )
                st.session_state.processing_thread.start()
                st.session_state.processing_start_time = time.time()

        # Display DeepMind results
        if st.session_state.get("deepmind_response"):
            st.subheader("💡 Cognitive Support Plan")
            st.markdown(st.session_state.deepmind_response)

        if st.session_state.get("deepmind_error"):
            st.error(st.session_state.deepmind_error)

        # Cognitive Load Analysis
        st.subheader("📈 Cognitive Load Analysis")
        if st.button("🧪 Run Cognitive Load Analysis"):
            with st.spinner("Analyzing cognitive load..."):
                analysis = deep_agent.get_cognitive_load_analysis()

                if "error" in analysis:
                    st.error(f"Analysis failed: {analysis['error']}")
                else:
                    st.session_state.cognitive_analysis = analysis

        if st.session_state.get("cognitive_analysis"):
            analysis = st.session_state.cognitive_analysis
            level = analysis.get("cognitive_load_level", "Unknown")
            explanation = analysis.get("explanation", "No explanation")

            # Display with color coding
            if level == "Low":
                st.success(f"**Cognitive Load Level:** 🟢 {level}")
            elif level == "Medium":
                st.warning(f"**Cognitive Load Level:** 🟡 {level}")
            elif level == "High":
                st.error(f"**Cognitive Load Level:** 🔴 {level}")
            else:
                st.info(f"**Cognitive Load Level:** {level}")

            st.info(f"**Explanation:** {explanation}")

        # Teacher Feedback
        with st.expander("📄 Teacher Feedback", expanded=False):
            teacher_feedback_ui()

    # Processing status and cancellation
    if st.session_state.deepmind_processing:
        if st.session_state.processing_thread and st.session_state.processing_thread.is_alive():
            st.info("⏳ Generating cognitive support plan... Please wait.")
        else:
            try:
                result = st.session_state.result_queue.get_nowait()
                st.session_state.deepmind_response = result
                st.session_state.deepmind_processing = False
                st.success("✅ Cognitive support plan generated!")
                logger.info("DeepMind solution generated successfully")
            except queue.Empty:
                st.info("⏳ Waiting for processing results...")

    # Save chat messages to ChromaDB
    if st.session_state.deepmind_response and st.session_state.current_user_input:
        save_chat_to_chroma(st.session_state.current_user_input, st.session_state.deepmind_response)
        # Clear current input after saving
        st.session_state.current_user_input = None


if __name__ == "__main__":
    run_chatbot()
