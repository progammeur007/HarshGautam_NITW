# main.py
# main.py

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from google import genai
from google.genai import types
from google.genai.errors import APIError
import requests
import io
import os
import json 
import logging
from typing import List, Dict, Any, Union

# Robust OCR/Image Libraries
import pytesseract
from PIL import Image
import cv2
import numpy as np

# --- FIX: Import load_dotenv ---
from dotenv import load_dotenv 
# ------------------------------

# Import all models
from models import (
    ExtractionRequest, ExtractionResponse, BillData, PagewiseLineItems, 
    TokenUsage, FinalTotalKVP, PagewiseListRoot 
)

# --- 1. CRITICAL TESSERACT CONFIGURATION & LOGGING ---
# Set the path to the Tesseract executable. THIS MUST BE CORRECT.
try:
    # --- REPLACE THIS PATH WITH YOUR ACTUAL TESSERACT.EXE LOCATION ---
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    # Allows deployment environments (like Render) that lack local Tesseract to skip this step.
    pass 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("BFHL_Extractor")

load_dotenv() # <--- THIS LINE WILL NOW WORK
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- FastAPI App Setup ---
app = FastAPI(title="BFHL Datathon Bill Extraction API", version="1.0.0")


# --- TOKEN TRACKER CLASS (Unchanged) ---
class TokenTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
    def add_usage(self, usage_metadata: Union[dict, types.UsageMetadata]):
        if isinstance(usage_metadata, types.UsageMetadata):
            self.input_tokens += usage_metadata.prompt_token_count
            self.output_tokens += usage_metadata.candidates_token_count
        elif isinstance(usage_metadata, dict):
            self.input_tokens += usage_metadata.get('prompt_token_count', 0)
            self.output_tokens += usage_metadata.get('candidates_token_count', 0)
    def get_token_usage(self) -> TokenUsage:
        return TokenUsage(
            total_tokens=self.input_tokens + self.output_tokens,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens
        )
# --- LLM Client Initialization ---
try:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found. API calls will fail.")
        client = None 
    else:
        client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Error initializing Gemini Client: {e}", exc_info=True)
    client = None


def fetch_and_ocr(document_url: str) -> str:
    """ROBUST OCR LAYER: Downloads, preprocesses (cleaning), and extracts text via Tesseract."""
    logger.info(f"Starting robust OCR for: {document_url}")
    
    if not hasattr(pytesseract.pytesseract, 'tesseract_cmd'):
        logger.warning("Tesseract path not set. Skipping preprocessing and returning fail code.")
        return "OCR_FAILED_GENERIC_ERROR" 
    
    try:
        doc_response = requests.get(document_url, timeout=30)
        doc_response.raise_for_status()

        image_bytes = io.BytesIO(doc_response.content)
        img_pil = Image.open(image_bytes).convert('RGB')
        
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        extracted_text = pytesseract.image_to_string(
            thresh, 
            lang='eng', 
            config='--psm 6' 
        )
        
        if not extracted_text.strip():
            logger.warning("Tesseract returned empty text.")
            return "OCR_FAILED_TEXT_EMPTY" 
        
        logger.info("OCR successful. Extracted text length: %d", len(extracted_text))
        return extracted_text

    except Exception as e:
        logger.error(f"ROBUST OCR LAYER FAILED: {e}", exc_info=True)
        return "OCR_FAILED_GENERIC_ERROR"


def extract_json_from_response(response: types.GenerateContentResponse) -> Any:
    """CRITICAL FIX: Manually extracts the clean JSON string from the response payload."""
    try:
        raw_json_string = response.candidates[0].content.parts[0].text
        return json.loads(raw_json_string)
    except Exception:
        logger.error("LLM returned non-compliant or unreadable JSON.", exc_info=True)
        raise HTTPException(status_code=500, detail="LLM returned non-compliant or unreadable JSON.")


@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: ExtractionRequest):
    
    logger.info(f"--- SUBMISSION CHECK STARTED for {request.document[:60]}... ---")
    
    if not client:
        raise HTTPException(status_code=503, detail="LLM service unavailable. Check API Key configuration.")

    tracker = TokenTracker()
    
    try:
        # Step 1: Run the robust OCR pipeline to get clean text
        extracted_text = fetch_and_ocr(request.document)
        
        if extracted_text == "OCR_FAILED_GENERIC_ERROR":
             raise HTTPException(status_code=500, detail="Document processing failed in the robust OCR layer.")

        logger.info(f"OCR Snippet: {extracted_text[:200].replace('\n', ' ')}...")
        
        # --- CALL 1: Line Items and Page Classification (Semantic Parsing) ---
        prompt_line_items = (
            "You are an expert financial document extractor specializing in handwritten, multilingual bills. "
            "Your output MUST strictly adhere to the provided Pydantic JSON schema (a list of PagewiseLineItems). "
            "Analyze the following text extracted via OCR. GUARDRAILS: ONLY extract numeric values clearly associated with currency or price for item_amount/item_rate. DO NOT extract Sl#, Cpt Code, or Date into any amount field. TEXT: "
        ) + extracted_text 
        
        response_line_items = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_line_items], 
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PagewiseListRoot 
            )
        )
        logger.info(f"Call 1 (Line Items) completed. Tokens added: {response_line_items.usage_metadata.total_token_count}")
        
        if response_line_items.usage_metadata:
            tracker.add_usage(response_line_items.usage_metadata)
            
        extracted_object_1 = extract_json_from_response(response_line_items)
        pagewise_root = PagewiseListRoot.model_validate(extracted_object_1)
        llm_output_list: List[PagewiseLineItems] = pagewise_root.root 

        # --- CALL 2: KVP Extraction for Actual Bill Total (Reconciliation Target) ---
        prompt_totals = (
            "You are an expert auditor. From the document, extract ONLY the final total amount "
            "(the authoritative Grand Total, Net Payable Amount, or Final Total) and the Sub-total (if explicitly present). "
            "STRICTLY return the data in the Pydantic JSON schema (FinalTotalKVP). TEXT: "
        ) + extracted_text

        response_totals = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_totals],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FinalTotalKVP
            )
        )
        
        logger.info(f"Call 2 (KVP Totals) completed. Tokens added: {response_totals.usage_metadata.total_token_count}")
        
        if response_totals.usage_metadata:
            tracker.add_usage(response_totals.usage_metadata)
            
        extracted_totals_object = extract_json_from_response(response_totals)
        extracted_totals: FinalTotalKVP = FinalTotalKVP.model_validate(extracted_totals_object)

        # --- 3. Final Reconciliation and Validation ---
        all_line_items = [item for page in llm_output_list for item in page.bill_items]
        total_item_count = len(all_line_items) 
        calculated_reconciled_amount = sum(item.item_amount for item in all_line_items)

        # CRITICAL ACCURACY LOGGING
        logger.info("--- ACCURACY CHECK ---")
        logger.info(f"Actual Bill Total (KVP Extraction): {extracted_totals.actual_bill_total}")
        logger.info(f"Calculated Sum (Line Item Sum): {round(calculated_reconciled_amount, 2)}")
        logger.info(f"Total Tokens Used: {tracker.get_token_usage().total_tokens}")
        logger.info("----------------------")

        # 4. Final Response Construction
        final_data = BillData(
            pagewise_line_items=llm_output_list,
            total_item_count=total_item_count,
        )
        
        logger.info("Submission check processed successfully.")
        return ExtractionResponse(
            is_success=True,
            token_usage=tracker.get_token_usage(),
            data=final_data
        )

    except APIError as e:
        logger.error(f"Gemini API Error during extraction: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM API Error: {e}")
    except ValidationError as e:
        logger.error(f"Pydantic Validation Error (Schema mismatch): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM output validation error. Model output did not match schema.")
    except Exception as e:
        logger.error(f"UNCAUGHT EXCEPTION: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")