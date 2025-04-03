import logging
import os

from . import input_processor
from . import notebook_builder
from . import prompt_builder
from . import ai_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OrchestrationError(Exception):
    pass

def run_generation_pipeline (
        csv_file_path : str,
        pdf_file_path: str,
        config: dict,

        ipynb_file_path: str | None = None,
        user_goal: str | None = None,
) -> str:
    
    logging.info("Starting notebook generation pipeline...")

    if not os.path .exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not Found!: {csv_file_path}")
    if not os.path .exists(pdf_file_path):
        raise FileNotFoundError(f"PDF file not Found!: {pdf_file_path}")
    if ipynb_file_path and not os.path .exists(ipynb_file_path):
        raise FileNotFoundError(f"IPYNB file not Found!: {ipynb_file_path}")
    
    if not config.get('GEMINI_MODEL_NAME'):
        raise OrchestrationError("Configuration missing 'GEMINI_MODEL_NAME'")
    

    try:
        logging.info(f'Processing Csv file : {csv_file_path}')
        csv_summary = input_processor.process_csv(csv_file_path)
        logging.info(f'CSV processing successful.')
        
        logging.info(f"Processing PDF: {pdf_file_path}")
        pdf_text = input_processor.process_pdf(pdf_file_path)
        logging.info("PDF processing successful.")

        ipynb_context = None
        if ipynb_file_path:
            logging.info(f'Processing ipynb file: {ipynb_file_path}')
            ipynb_context = input_processor.process_ipynb(ipynb_file_path)
            logging.info('IPYNB processing successful.')
        else:
            logging.info('No ipynb file provided')
    except Exception as e:
        logging.info(f"Error during input processing: {e}", exc_info=True)      
        raise OrchestrationError(f"Failed to process input files: {e}") from e
  
    
    #    --Build prompt--
    try:
        logging.info("Building prompt for AI model!")
        prompt = prompt_builder.build_generation_prompt(
            csv_summary=csv_summary, 
            pdf_text=pdf_text,
            user_goal = user_goal or "Perform standard Exploratory Data Analysis (EDA) and suggest next steps.",
            ipynb_context = ipynb_context,
        )
        logging.info("Prompt built successfully.")
    except Exception as e:
        logging.info(f"Error during prompt building: {e}", exc_info=True)
        raise OrchestrationError(f"Failed to build prompt: {e}") from e
    

    # --Call Ai--
    try:
        logging.info(f"calling Ai model in our case we use gemini {config['GEMINI_MODEL_NAME']}")
        raw_ai_response = ai_client.get_gemini_response(
            prompt = prompt,
            api_key =config['GEMINI_API_KEY'],
            model_name = config['GEMINI_MODEL_NAME']  
        )

        if not raw_ai_response:
            raise OrchestrationError("Received empty response from AI model.")
        logging.info("AI response received successfully.")

    except Exception as e:
        logging.info(f"Error during AI call: {e}", exc_info=True)
        raise OrchestrationError(f"Failed to get response from AI: {e}") from e
    
    # --Build Notebook--

    try:
        logging.info("Building .ipynb file from AI response!")
        notebook_json_string = notebook_builder.create_ipynb_from_ai_response(raw_ai_response)
        logging.info(".ipynb file built successfully.")
    except Exception as e:
        logging.error(f"Error during notebook building: {e}", exc_info=True)
        raise OrchestrationError(f"Failed to construct notebook from AI response: {e}") from e
    
    # --return result--
    logging.info("Notebook generation pipeline completed successfully.")
    return notebook_json_string