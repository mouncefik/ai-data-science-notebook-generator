# agent/notebook_builder.py

import nbformat
import logging
import re # Using regex for more robust splitting

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants should match those used in prompt_builder.py
MARKDOWN_TAG = "[MARKDOWN]"
CODE_TAG = "[CODE]"

class NotebookBuilderError(Exception):
    """Custom exception for errors during notebook building."""
    pass

def create_ipynb_from_ai_response(ai_response_text: str) -> str:
    """
    Parses the AI's text response (with [MARKDOWN] and [CODE] tags)
    and creates a Jupyter Notebook (.ipynb) JSON string.

    Args:
        ai_response_text: The raw string response from the Gemini API.

    Returns:
        A JSON string representing the generated .ipynb file.

    Raises:
        NotebookBuilderError: If the AI response cannot be parsed or the notebook
                              cannot be created.
    """
    logging.info("Starting notebook construction from AI response...")

    if not ai_response_text or not ai_response_text.strip():
        logging.error("AI response text is empty or whitespace only.")
        raise NotebookBuilderError("Cannot build notebook from empty AI response.")

    # Create a new notebook object (using schema version 4)
    notebook = nbformat.v4.new_notebook()

    # Use regex to split the text by the tags, keeping the tags as delimiters
    # This pattern looks for the tags at the beginning of a line (^),
    # optionally preceded by whitespace (\s*), and captures the tag itself.
    # It splits the text based on these occurrences.
    # Pattern explanation:
    # ^               - Start of the line
    # \s*             - Optional whitespace
    # (\\[(?:MARKDOWN|CODE)\\]) - Capture group 1: Literal '[' + 'MARKDOWN' or 'CODE' + literal ']'
    # \s*             - Optional whitespace following the tag (including newline)
    pattern = re.compile(r"^\s*(\[(?:MARKDOWN|CODE)\])\s*", re.MULTILINE)
    parts = pattern.split(ai_response_text)

    # Debug: Print the split parts
    # logging.debug(f"Split parts: {parts}")

    current_cell_type = None
    cell_content = ""

    # The first part before any tag is usually empty or unwanted preamble, skip it if empty
    start_index = 1 if not parts[0].strip() else 0

    if start_index >= len(parts):
         logging.error("AI response does not seem to contain valid tags or content after splitting.")
         raise NotebookBuilderError("Could not find valid content sections after tags in AI response.")


    for i in range(start_index, len(parts)):
        part = parts[i]

        if part == MARKDOWN_TAG:
            if current_cell_type and cell_content.strip():
                # Add the previous cell before starting the new one
                add_cell(notebook, current_cell_type, cell_content.strip())
            current_cell_type = 'markdown'
            cell_content = "" # Reset content for the new cell
            logging.debug("Found MARKDOWN tag.")
        elif part == CODE_TAG:
            if current_cell_type and cell_content.strip():
                 # Add the previous cell before starting the new one
                add_cell(notebook, current_cell_type, cell_content.strip())
            current_cell_type = 'code'
            cell_content = "" # Reset content for the new cell
            logging.debug("Found CODE tag.")
        else:
            # This part is the content following a tag
            if current_cell_type:
                cell_content += part
            else:
                # Content before the first valid tag - log it but usually ignore
                if part.strip():
                     logging.warning(f"Ignoring content found before the first valid tag: '{part[:100]}...'")


    # Add the last cell content if any exists
    if current_cell_type and cell_content.strip():
        add_cell(notebook, current_cell_type, cell_content.strip())

    if not notebook.cells:
        logging.error("No cells were added to the notebook. Check AI response format and tags.")
        raise NotebookBuilderError("Failed to parse any valid cells from the AI response.")

    # --- Validate and Serialize ---
    try:
        # nbformat.validate(notebook) # Optional: Strict validation (can be too strict sometimes)
        notebook_json_string = nbformat.writes(notebook)
        logging.info(f"Notebook construction successful. Created {len(notebook.cells)} cells.")
        return notebook_json_string
    except Exception as e:
        logging.error(f"Failed to serialize or validate the notebook object: {e}", exc_info=True)
        raise NotebookBuilderError(f"Error writing notebook object: {e}") from e


def add_cell(notebook, cell_type: str, content: str):
    """Helper function to create and add a cell to the notebook object."""
    logging.debug(f"Adding {cell_type} cell. Content length: {len(content)}")
    if cell_type == 'markdown':
        cell = nbformat.v4.new_markdown_cell(content)
    elif cell_type == 'code':
        # Basic cleaning: remove potential leading/trailing whitespace/newlines
        # that might be artifacts of the splitting/joining process.
        cleaned_content = content.strip()
        cell = nbformat.v4.new_code_cell(cleaned_content)
    else:
        logging.warning(f"Unknown cell type encountered: {cell_type}. Skipping.")
        return # Don't add unknown cell types

    notebook.cells.append(cell)

