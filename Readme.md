# AI Data Science Notebook Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Generate initial data science Jupyter Notebooks (`.ipynb`) automatically. Provide your data (CSV), a description of the data (PDF), and optionally an existing notebook for context, and let the AI kickstart your analysis workflow.

This tool aims to reduce the boilerplate code often required for initial data loading, cleaning, and exploratory data analysis (EDA), allowing data scientists to focus on deeper insights faster.

## Demo

Watch the demo video: [demo.webm](demo/Demo.webm)

## Features

*   **CSV Data Input:** Upload your primary dataset in CSV format.
*   **PDF Data Description:** Provide context about your data columns, meanings, and potential issues via a PDF document (e.g., a data dictionary).
*   **Optional IPYNB Context:** Upload an existing Jupyter Notebook (`.ipynb`) to give the AI context about libraries you prefer or previous steps taken.
*   **User Goal Input:** Specify a high-level goal for the analysis (e.g., "Perform EDA", "Build a churn model").
*   **AI-Powered Generation:** Leverages the Google Gemini API (configurable model) to generate notebook content.
*   **Structured Output:** Generates a complete `.ipynb` file containing both Markdown explanation cells and Python code cells.
*   **Standard Workflow:** Follows a typical data science workflow structure (Setup, Load, Clean, EDA, etc.).
*   **Artifact Generation:** Includes code snippets within the generated notebook to save outputs like plots (`.png`) or models (`.pkl`) where appropriate.
*   **Streamlit Frontend:** Easy-to-use web interface built with Streamlit.
*   **Downloadable Output:** Download the generated `.ipynb` file directly from the interface.

## Prerequisites

*   **Python:** Version 3.9 or higher recommended.
*   **pip:** Python package installer (usually included with Python).
*   **Google Gemini API Key:** You need an API key from Google API.
*   **Input Files:**
    *   A dataset in `.csv` format.
    *   A `.pdf` file describing the columns and data in the CSV file.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Mouncefik/ai-data-science-notebook-generator.git 
    cd ai-notebook-generator
    ```

2.  **Create a Virtual Environment (Recommended):**
    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application requires your Google Gemini API key. There are two ways to provide it:

1.  **`.env` File (Recommended):**
    *   Create a file named `.env` in the root directory of the project (`ai-notebook-generator/`).
    *   Add your API key to this file:
        ```dotenv
        # .env
        GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_HERE
        ```
    *   The application will automatically load the key from this file using `python-dotenv`.

2.  **Streamlit UI:**
    *   If the `.env` file is not found or the key is missing, the Streamlit application sidebar will prompt you to enter your API key directly.

## How to Run

1.  Make sure your virtual environment is activated.
2.  Ensure you are in the root directory of the project (`ai-notebook-generator/`).
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application should open automatically in your web browser.

## Project Structure

```
ai-notebook-generator/
│
├── app.py                 # Main Streamlit application file (Frontend + UI Logic)
│
├── agent/                 # Core backend logic package
│   ├── __init__.py        # Makes 'agent' a Python package
│   ├── orchestrator.py    # Coordinates the generation pipeline workflow
│   ├── input_processor.py # Functions for parsing PDF, CSV, IPYNB inputs
│   ├── prompt_builder.py  # Functions to construct the prompt for the Gemini API
│   ├── ai_client.py       # Functions to interact with the Google Gemini API
│   └── notebook_builder.py# Functions using nbformat to create the final .ipynb file
│
├── .env                   # Stores API keys and potentially other secrets (!!! DO NOT COMMIT THIS FILE !!!)
├── requirements.txt       # List of Python dependencies for pip
├── README.md              # This file
│
└── outputs/               # (Optional) Can be used for temporary storage or logging outputs
```

## How to Use the Application

1.  **Launch the App:** Run `streamlit run app.py`.
2.  **Configure API Key:** Ensure your API key is loaded from `.env` or enter it in the sidebar.
3.  **Select Model:** Choose the Gemini model you wish to use (e.g., `gemini-1.5-flash`).
4.  **Upload CSV:** Use the file uploader in the sidebar to upload your `.csv` data file.
5.  **Upload PDF:** Upload the corresponding `.pdf` data description file.
6.  **Upload IPYNB (Optional):** If you have an existing notebook to provide context, upload it.
7.  **Enter Goal (Optional):** Describe the main objective of the analysis you want the notebook to perform.
8.  **Generate:** Click the "✨ Generate Notebook" button.
9.  **Wait:** The application will process the inputs and call the Gemini API. This might take a few moments depending on file sizes and API response time. A spinner will indicate progress.
10. **Download:** Once generation is complete, a "⬇️ Download Generated Notebook (.ipynb)" button will appear. Click it to save the file.
11. **Review:** **Crucially, always open and carefully review the generated `.ipynb` file. Verify the code and logic before executing any cells, especially those involving data modification or model training.**

## Potential Enhancements

*   Support for other data formats (e.g., Parquet, Excel, JSON).
*   More sophisticated PDF parsing to extract structured information like tables.
*   Smarter context extraction from optional `.ipynb` input (e.g., identifying key variables or functions).
*   Option to directly execute generated code in a sandboxed environment (with strong security warnings).
*   Caching of results for identical inputs.
*   More granular control over the generation process (e.g., specifying libraries, choosing analysis steps).
*   Improved error handling and user feedback.
*   Support for multi-turn conversations to refine the notebook.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file (if you create one) or the badge at the top for details.
```