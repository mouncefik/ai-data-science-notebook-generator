import logging
import json 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MARKDOWN_TAG = "[MARKDOWN]"
CODE_TAG = "[CODE]"

def format_csv_summary(csv_summary: dict) -> str:
    if not csv_summary:
        return "No CSV summary provided."

    summary_parts = [
        f"- **File Name:** `{csv_summary.get('file_name', 'N/A')}`",
        f"- **Shape:** {csv_summary.get('shape', 'N/A')} (rows, columns)",
        f"- **Columns:** {', '.join(csv_summary.get('columns', []))}",
        f"- **Data Types Summary:**\n```\n{csv_summary.get('dtypes_summary', 'N/A')}\n```",
        f"- **Data Preview (First few rows):**\n```\n{csv_summary.get('head_preview', 'N/A')}\n```",
        f"- **Descriptive Statistics:**\n```\n{csv_summary.get('description_stats', 'N/A')}\n```",
        f"- **Missing Values Summary:**\n```\n{csv_summary.get('missing_values_summary', 'N/A')}\n```"
    ]
    return "\n".join(summary_parts)

def format_ipynb_context(ipynb_context: dict | None) -> str:
    if not ipynb_context:
        return "No existing notebook context provided."
    return ipynb_context.get('message', 'Could not parse IPYNB context.')


def build_generation_prompt(
    csv_summary: dict,
    pdf_text: str,
    ipynb_context: dict | None = None,
    user_goal: str | None = None
    ) -> str:
    
    logging.info("Building generation prompt...")

    # --- Define the AI's Role and Task ---
    role_and_task = f"""You are an expert Python data scientist AI assistant. Your task is to generate a complete Jupyter Notebook (.ipynb) file content based on the provided data summary, data description, and user goal.

The output MUST be a single block of text containing alternating Markdown and Python code cells, clearly delimited by `{MARKDOWN_TAG}` and `{CODE_TAG}` respectively.
Example:
{MARKDOWN_TAG}
# Notebook Title
This is an introductory markdown cell.
{CODE_TAG}
import pandas as pd
import numpy as np
print("Libraries imported.")
{MARKDOWN_TAG}
## Load Data
Now, we load the data.
{CODE_TAG}
# Code to load data goes here...

Follow these instructions precisely:
1.  **Structure:** Generate a logical flow for a data science task: Setup -> Load Data -> Data Cleaning/Preparation -> Exploratory Data Analysis (EDA) -> Feature Engineering (if applicable/needed) -> Modeling (if requested or appropriate) -> Conclusion/Summary.
2.  **Content:** Use the provided CSV Summary and PDF Description to understand the data and guide your analysis. Reference column names accurately.
3.  **Code:** Write clean, runnable Python code using standard libraries (pandas, numpy, matplotlib, seaborn, scikit-learn). Add comments to explain complex code sections. Assume the primary data file (details below) is available in the execution environment as '{csv_summary.get('file_name', 'data.csv')}'. **Crucially**, make sure the *first* code block imports necessary libraries.
4.  **Markdown:** Use Markdown cells effectively to explain the steps, observations, and rationale behind the code.
5.  **Artifacts:** Where appropriate (especially for EDA plots or final datasets/models), include Python code to SAVE the output to a file (e.g., `plt.savefig('plot_name.png')`, `df.to_csv('processed_data.csv')`, `joblib.dump(model, 'model.pkl')`). Print a confirmation message after saving (e.g., `print("Plot saved to plot_name.png")`).
6.  **Formatting:** Start with a `{MARKDOWN_TAG}` cell for the title. Ensure every Markdown section starts exactly with `{MARKDOWN_TAG}` on a new line and every code section starts exactly with `{CODE_TAG}` on a new line. Do NOT include any other text before or after these tags on their respective lines.
7.  **Completeness:** Generate the full notebook content in one continuous response. Do not add introductory or concluding remarks outside the tagged cell structure.
"""

    # --- Format Input Information ---
    formatted_csv_summary = format_csv_summary(csv_summary)
    formatted_pdf_text = pdf_text if pdf_text else "No data description provided."
    formatted_ipynb_context = format_ipynb_context(ipynb_context)
    final_user_goal = user_goal if user_goal else "Perform a comprehensive Exploratory Data Analysis (EDA) and provide insights."

    # --- Assemble the Final Prompt ---
    prompt = f"""{role_and_task}

--- INPUT DATA CONTEXT ---

**1. CSV Data Summary:**
{formatted_csv_summary}

**2. Data Description (from PDF):**
```text
{formatted_pdf_text}
```

**3. Existing Notebook Context (Optional):**
{formatted_ipynb_context}

**4. User Goal:**
{final_user_goal}

--- REQUIRED NOTEBOOK OUTPUT ---
Generate the notebook content now, starting with `{MARKDOWN_TAG}`:
"""

    logging.info("Prompt built successfully.")
    return prompt
