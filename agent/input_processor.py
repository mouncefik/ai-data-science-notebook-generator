import pandas as pd
import logging
import io

import nbformat
import PyPDF2


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Csv Processing ---

def process_csv(csv_file_path: str, max_row_preview: int = 5) -> dict:
    logging.info("processing csv!")
    try:
        df = pd.read_csv(csv_file_path)
        logging.info("csv processed!")

        # --extract info

        shape = df.shape
        columns = df.columns.tolist()


        # --get dtypes as string

        dtypes_buffer = io.StringIO()
        df.info(buf=dtypes_buffer)
        dtypes_string = dtypes_buffer.getvalue()

        head_string  = df.head(max_row_preview).to_string()

        description_string = df.describe(include='all').to_string()

        # -- missing value as string
        missing_values = df.isnull().sum()
        missing_values_string = missing_values[missing_values > 0].to_string()

        if not missing_values_string.strip() or "Empty" in missing_values_string:
            missing_values_string = "No missing Values found!"

        
        summary = {
            'file_name': csv_file_path.split('/')[-1].split('\\')[-1],
            'shape': shape,
            'columns': columns,
            'dtypes_summary': dtypes_string, 
            'head_preview': head_string,
            'description_stats': description_string,
            'missing_values_summary': missing_values_string
        }
        logging.info(f"Successfully processed CSV: {csv_file_path}. Shape={shape}")
        return summary
    
    except Exception as e:
        logging.error(f"An Error occured {csv_file_path}: {e}")
        raise Exception(f"Error processing {csv_file_path}: {e}") from e
    

# --- process pdf ---

def process_pdf(pdf_file_path: str) -> str:
    logging.info("processing pdf!")

    try:
        with open(pdf_file_path, 'rb') as f:
            pass
        return f"Placholder: content from {pdf_file_path} would be extracted here!"
    
    except FileNotFoundError:
        logging.error(f'PDF not found!! {pdf_file_path}')
    except Exception as e:
        logging.error(f"An Error occured {pdf_file_path}: {e}")
        raise Exception(f"Error processing {pdf_file_path}: {e}") from e

# --- process ipynb ---
def process_ipynb(ipynb_file_path: str) -> dict:
    logging.info("processing ipynb!")
    try:
        with open(ipynb_file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

            num_cells = len(nb.cells)
        return {"message": f"Placeholder: Found {num_cells} cells in {ipynb_file_path}. COntext extraction logic needed."}
    except FileNotFoundError:
        logging.error(f'IPython notebook not found!! {ipynb_file_path}')
    except Exception as e:
        logging.error(f"Error processing IPYNB {ipynb_file_path}: {e}", exc_info=True)
        raise Exception(f"Could not process IPYNB: {e}") from e