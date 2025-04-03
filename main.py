import streamlit as st
import os
import tempfile # To handle uploaded files safely
from dotenv import load_dotenv
import logging

# Import the orchestrator from our agent package
from agent import orchestrator
from agent.orchestrator import OrchestrationError # Import specific exception

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv() # Load environment variables from .env file

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Notebook Generator")
st.title("🤖 AI Data Science Notebook Generator")
st.markdown("""
Upload your data (CSV), provide a description (PDF), and optionally an existing notebook for context.
The AI will generate a new Jupyter Notebook (`.ipynb`) to kickstart your analysis.
""")

# --- Session State Initialization ---
# Store generated content and API key across reruns
if 'generated_notebook_content' not in st.session_state:
    st.session_state.generated_notebook_content = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "") # Initialize from .env


# --- Sidebar for Inputs and Configuration ---
with st.sidebar:
    st.header("Configuration")

    # API Key Input - prioritize .env, allow user override/input
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        st.success("✅ Gemini API Key loaded from .env")
        st.session_state.gemini_api_key = env_key
        st.session_state.api_key_valid = True
    else:
        st.warning("Gemini API Key not found in .env file.")
        user_api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Get your key from Google AI Studio: https://aistudio.google.com/app/apikey"
        )
        if user_api_key:
            st.session_state.gemini_api_key = user_api_key
            st.session_state.api_key_valid = True # Assume valid for now, client will verify
            st.info("API Key entered.")
        else:
            st.session_state.api_key_valid = False


    # Model Selection (Add more models as needed/available)
    model_name = st.selectbox(
        "Select Gemini Model:",
        ("gemini-2.0-flash", "gemini-2.5-pro-experimental-03-25"), # Add other compatible models if desired
        index=0 # Default to flash
    )

    st.header("Inputs")
    uploaded_csv = st.file_uploader("1. Upload Data File (.csv)", type=['csv'])
    uploaded_pdf = st.file_uploader("2. Upload Data Description (.pdf)", type=['pdf'])
    uploaded_ipynb = st.file_uploader("3. Upload Existing Notebook (Optional, .ipynb)", type=['ipynb'])

    user_goal = st.text_area(
        "4. Describe your Goal (Optional):",
        placeholder="e.g., 'Perform EDA and visualize correlations', 'Build a classification model for churn'",
        height=100
    )

    # --- Generate Button ---
    # Disable button if essential inputs are missing
    required_inputs_present = uploaded_csv and uploaded_pdf and st.session_state.api_key_valid
    generate_button = st.button(
        "✨ Generate Notebook",
        type="primary",
        disabled=not required_inputs_present,
        help="Requires CSV, PDF, and a valid API Key to be set."
    )
    if not required_inputs_present:
        if not st.session_state.api_key_valid:
             st.warning("Please enter your Gemini API Key.")
        if not uploaded_csv:
             st.warning("Please upload a CSV file.")
        if not uploaded_pdf:
             st.warning("Please upload a PDF file.")


# --- Main Area for Output ---

# Reset status on new interaction before potentially running the pipeline
st.session_state.error_message = None
# Don't reset generated_notebook_content here, keep it until next successful generation or clear explicitly

if generate_button and required_inputs_present:
    st.session_state.generated_notebook_content = None # Clear previous result before new run

    # Use temporary files to store uploaded data for the orchestrator
    # Using 'with' ensures files are cleaned up automatically
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            tmp_csv.write(uploaded_csv.getvalue())
            tmp_csv_path = tmp_csv.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(uploaded_pdf.getvalue())
            tmp_pdf_path = tmp_pdf.name

        tmp_ipynb_path = None
        if uploaded_ipynb:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp_ipynb:
                tmp_ipynb.write(uploaded_ipynb.getvalue())
                tmp_ipynb_path = tmp_ipynb.name

        # Prepare configuration for the orchestrator
        config = {
            'GEMINI_API_KEY': st.session_state.gemini_api_key,
            'GEMINI_MODEL_NAME': model_name
        }

        logging.info("Starting orchestrator pipeline via Streamlit...")
        with st.spinner(f"🚀 Generating notebook using {model_name}... This may take a moment."):
            try:
                # Call the main function of the orchestrator
                generated_json = orchestrator.run_generation_pipeline(
                    csv_file_path=tmp_csv_path,
                    pdf_file_path=tmp_pdf_path,
                    config=config,
                    ipynb_file_path=tmp_ipynb_path, # Will be None if no file uploaded
                    user_goal=user_goal
                )
                st.session_state.generated_notebook_content = generated_json
                st.success("✅ Notebook generated successfully!")
                logging.info("Orchestrator pipeline completed successfully.")

            except (OrchestrationError, FileNotFoundError, ValueError) as e:
                st.session_state.error_message = f"Pipeline Error: {e}"
                logging.error(f"Orchestration failed: {e}", exc_info=True)
            except Exception as e: # Catch any other unexpected errors
                 st.session_state.error_message = f"An unexpected error occurred: {e}"
                 logging.error(f"Unexpected error in pipeline: {e}", exc_info=True)

    finally:
        # Ensure temporary files are deleted even if errors occur
        logging.debug("Cleaning up temporary files...")
        if 'tmp_csv_path' in locals() and os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)
            logging.debug(f"Removed temp CSV: {tmp_csv_path}")
        if 'tmp_pdf_path' in locals() and os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)
            logging.debug(f"Removed temp PDF: {tmp_pdf_path}")
        if tmp_ipynb_path and os.path.exists(tmp_ipynb_path):
            os.remove(tmp_ipynb_path)
            logging.debug(f"Removed temp IPYNB: {tmp_ipynb_path}")


# --- Display Results or Errors ---
st.subheader("Output")

# Display errors if they occurred
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# Display download button if content was generated successfully
if st.session_state.generated_notebook_content:
    # Determine a safe filename based on the uploaded CSV name
    base_filename = "generated_notebook"
    if uploaded_csv:
        base_filename = os.path.splitext(uploaded_csv.name)[0] + "_analysis"

    st.download_button(
        label="⬇️ Download Generated Notebook (.ipynb)",
        data=st.session_state.generated_notebook_content,
        file_name=f"{base_filename}.ipynb",
        mime="application/x-ipynb+json", # Standard MIME type for notebooks
    )
    # Optional: Display a preview (e.g., first few cells - requires parsing the JSON)
    # For simplicity, we'll just offer the download for now.

elif not st.session_state.error_message and not generate_button:
    st.info("Upload files and click 'Generate Notebook' to start.")


# --- Footer/Instructions ---
st.markdown("---")
st.markdown("""
**Notes:**
- Ensure your API key is correctly entered or present in your `.env` file.
- Larger files or complex goals may take longer to process.
- The quality of the generated notebook depends heavily on the data description quality and the AI model's capabilities. **Always review generated code before execution.**
""")