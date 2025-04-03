import google.generativeai as genai
import logging
import os
import time
from google.api_core import exceptions as google_exceptions # Import specific exceptions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.4,
    "top_p": 1.0, # Keep high for code gen? Or lower? Let's start high.
    "top_k": 32, # Common default
    "max_output_tokens": 8192, # Max allowed by many models, adjust if needed
}

DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

class AIClientError(Exception):
    pass


_gemini_configured = False

def get_gemini_response(
    prompt: str,
    api_key: str,
    model_name: str,
    generation_config_override: dict | None = None,
    safety_settings_override: list | None = None,
    max_retries: int = 2,
    initial_delay: float = 1.0
    ) -> str:
    
    global _gemini_configured

    if not api_key:
        raise ValueError("API key is required.")
    if not model_name:
        raise ValueError("Model name is required.")

    # --- Configure API Key 
    if not _gemini_configured:
        try:
            logging.info("Configuring Google Generative AI...")
            genai.configure(api_key=api_key)
            _gemini_configured = True
            logging.info("Google Generative AI configured successfully.")
        except Exception as e:
            logging.exception("Failed to configure Google Generative AI")
            raise AIClientError(f"Gemini API configuration failed: {e}") from e

    # --- Prepare Configuration
    gen_config = DEFAULT_GENERATION_CONFIG.copy()
    if generation_config_override:
        gen_config.update(generation_config_override)

    safety_settings = safety_settings_override if safety_settings_override is not None else DEFAULT_SAFETY_SETTINGS

    # --- Instantiate Model
    try:
        logging.info(f"Instantiating Gemini model: {model_name}")
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=gen_config,
            safety_settings=safety_settings
        )
        logging.info("Model instantiated successfully.")
    except Exception as e:
        logging.exception(f"Failed to instantiate model: {model_name}")
        raise AIClientError(f"Failed to create GenerativeModel instance: {e}") from e

    # --- Call API with Retries
    current_retry = 0
    delay = initial_delay
    while current_retry <= max_retries:
        try:
            logging.info(f"Sending prompt to Gemini model (Attempt {current_retry + 1}/{max_retries + 1})...")
            response = model.generate_content(prompt)

            logging.info("Received response from Gemini.")

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                logging.error(f"API call blocked by safety settings. Reason: {reason}")
                raise AIClientError(f"Content generation blocked due to safety settings: {reason}")

            if not response.candidates:
                 logging.error("API response received, but it contains no candidates.")
                 finish_reason = "Unknown (no candidates)"
                 if response.prompt_feedback and hasattr(response.prompt_feedback, 'finish_reason'):
                      finish_reason = response.prompt_feedback.finish_reason
                 raise AIClientError(f"AI response has no candidates. Generation may have been stopped (Reason: {finish_reason}) or blocked.")

            try:
                generated_text = response.text # This is a convenient shortcut
                if not generated_text or not generated_text.strip():
                    logging.warning("AI response has candidates but the extracted 'response.text' is empty.")
                    if response.candidates[0].content and response.candidates[0].content.parts:
                         generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))

                    if not generated_text or not generated_text.strip():
                         finish_reason = response.candidates[0].finish_reason if hasattr(response.candidates[0], 'finish_reason') else 'UNKNOWN'
                         logging.error(f"AI generated empty content. Finish reason: {finish_reason}")
                         raise AIClientError(f"AI returned empty content. Finish Reason: {finish_reason}")

                logging.info("Successfully extracted text from AI response.")
                return generated_text.strip() # Return cleaned text

            except (AttributeError, IndexError, ValueError) as e:
                 logging.exception("Error extracting text content from valid API response structure.")
                 raise AIClientError(f"Could not extract text from response: {e}") from e
            except StopIteration: # Handle case where response.text raises this if no text part exists
                 logging.error("AI response generated, but no text part found (response.text failed).")
                 raise AIClientError("AI response did not contain a text part.")


        except (google_exceptions.DeadlineExceeded,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError,
                google_exceptions.ResourceExhausted) as e: # ResourceExhausted could be rate limits
            logging.warning(f"API call failed with retryable error: {type(e).__name__}. Retrying in {delay:.2f}s...")
            if current_retry == max_retries:
                logging.error(f"API call failed after {max_retries} retries: {e}", exc_info=True)
                raise AIClientError(f"API call failed after {max_retries} retries: {e}") from e
            time.sleep(delay)
            current_retry += 1
            delay *= 2 # Exponential backoff

        except (google_exceptions.PermissionDenied, google_exceptions.Unauthenticated) as e:
             logging.error(f"API call failed due to authentication/permission error: {e}", exc_info=False) # Don't log full trace usually
             raise AIClientError(f"Authentication/Permission Error: {e}. Check your API key.") from e
        except google_exceptions.InvalidArgument as e:
             logging.error(f"API call failed due to invalid argument: {e}", exc_info=True) # Log trace here, might be bad prompt/config
             raise AIClientError(f"Invalid Argument Error: {e}. Check model name, prompt, or generation config.") from e
        except google_exceptions.NotFound as e:
             logging.error(f"API call failed because resource (e.g., model) was not found: {e}", exc_info=False)
             raise AIClientError(f"Model or resource not found: {e}. Check model name: '{model_name}'.") from e

        # --- Handle Other Unexpected Errors ---
        except Exception as e:
            logging.exception("An unexpected error occurred during the Gemini API call.")
            # Decide if this general exception should be retried or not
            # For now, treat unexpected errors as non-retryable immediately
            raise AIClientError(f"An unexpected error occurred: {e}") from e

    raise AIClientError("Exited retry loop unexpectedly without success or specific error.")

