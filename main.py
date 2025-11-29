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
from dotenv import load_dotenv
from typing import List, Dict, Any, Union

# Import all models
from models import (
    ExtractionRequest, 
    ExtractionResponse, 
    BillData, 
    PagewiseLineItems, 
    BillItem, 
    TokenUsage,
    FinalTotalKVP,
    PagewiseListRoot 
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="BFHL Datathon Bill Extraction API", version="1.0.0")

# --- TOKEN TRACKER CLASS ---
class TokenTracker:
    """Utility class to aggregate token usage across multiple LLM calls."""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add_usage(self, usage_metadata: Union[dict, types.UsageMetadata]):
        """FIX: Accesses attributes directly from the UsageMetadata object."""
        if isinstance(usage_metadata, types.UsageMetadata):
            self.input_tokens += usage_metadata.prompt_token_count
            self.output_tokens += usage_metadata.candidates_token_count
        elif isinstance(usage_metadata, dict):
            self.input_tokens += usage_metadata.get('prompt_token_count', 0)
            self.output_tokens += usage_metadata.get('candidates_token_count', 0)

    def get_token_usage(self) -> TokenUsage:
        """Returns the final TokenUsage Pydantic model."""
        return TokenUsage(
            total_tokens=self.input_tokens + self.output_tokens,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens
        )
# ---------------------------

# --- LLM Client Initialization ---
try:
    if not GEMINI_API_KEY:
        client = None 
    else:
        client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    client = None
# ---------------------------------


def fetch_document(document_url: str) -> List[types.Part]:
    """Fetches the document from the URL and converts it into a list of Parts for the LLM."""
    try:
        doc_response = requests.get(document_url, timeout=30)
        doc_response.raise_for_status()
        mime_type = doc_response.headers.get('Content-Type', 'image/jpeg') 
        document_part = types.Part.from_bytes(data=doc_response.content, mime_type=mime_type)
        return [document_part]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch document URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error preparing document for LLM.")


def extract_json_from_response(response: types.GenerateContentResponse) -> Any:
    """CRITICAL FIX: Manually extracts the clean JSON string from the response payload."""
    try:
        raw_json_string = response.candidates[0].content.parts[0].text
        return json.loads(raw_json_string)

    except (AttributeError, IndexError, json.JSONDecodeError) as e:
        print(f"Failed to parse LLM structured output. Error: {e}")
        raise HTTPException(status_code=500, detail="LLM returned non-compliant or unreadable JSON.")


@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: ExtractionRequest):
    """API endpoint to extract all required bill data using a structured LLM call."""
    if not client:
        raise HTTPException(status_code=503, detail="LLM service unavailable. Check API Key configuration.")

    tracker = TokenTracker()
    
    try:
        document_parts = fetch_document(request.document)
        
        # --- CALL 1: Line Items and Page Classification ---
        prompt_line_items = (
            "You are an expert financial document extractor specializing in multilingual and handwritten hospital bills. "
            "Your output MUST strictly adhere to the provided Pydantic JSON schema (a list of PagewiseLineItems). "
            "For EACH page in the document, perform the following two tasks: "
            "1. **CLASSIFY** the page type as 'Bill Detail', 'Final Bill', 'Pharmacy', or 'Other'. "
            "2. **EXTRACT ALL LINE ITEMS** from the table. "
            "   - **Guardrail (Hint #2):** For item_amount and item_rate, ONLY use numeric values from columns explicitly representing currency (e.g., 'Gross Amount', 'Net Amt', 'Rate'). **DO NOT extract Sl#, Cpt Code, or Date into any amount field.** "
            "   - **Constraint:** Ensure the item_amount is the final price post-discounts. "
            "   - **Exclusion:** Do NOT extract header rows, sub-total rows (like 'Category Total'), or any row without a valid quantity. "
        )

        response_line_items = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_line_items] + document_parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PagewiseListRoot 
            )
        )
        
        if response_line_items.usage_metadata:
            tracker.add_usage(response_line_items.usage_metadata)
            
        extracted_object_1 = extract_json_from_response(response_line_items)
        pagewise_root = PagewiseListRoot.model_validate(extracted_object_1)
        llm_output_list: List[PagewiseLineItems] = pagewise_root.root 

        # --- CALL 2: Key-Value Pair (KVP) Extraction for Actual Bill Total ---
        prompt_totals = (
            "You are an expert auditor. From the document, your SOLE task is to extract the **Actual Bill Total** "
            "(the authoritative Grand Total, Net Payable Amount, or Final Total) and the Sub-total (if explicitly present). "
            "STRICTLY return the data in the Pydantic JSON schema (FinalTotalKVP). Ignore all line items and descriptive text."
        )

        response_totals = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_totals] + document_parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FinalTotalKVP
            )
        )
        
        if response_totals.usage_metadata:
            tracker.add_usage(response_totals.usage_metadata)
            
        extracted_totals_object = extract_json_from_response(response_totals)
        extracted_totals: FinalTotalKVP = FinalTotalKVP.model_validate(extracted_totals_object)

        # --- 3. Final Reconciliation and Validation ---
        all_line_items = [item for page in llm_output_list for item in page.bill_items]
        total_item_count = len(all_line_items) 
        calculated_reconciled_amount = sum(item.item_amount for item in all_line_items)

        # CRITICAL ACCURACY LOGGING
        print("--- ACCURACY CHECK ---")
        print(f"Actual Bill Total (KVP Extraction): {extracted_totals.actual_bill_total}")
        print(f"Calculated Sum (Line Item Sum): {round(calculated_reconciled_amount, 2)}")
        print("----------------------")

        # 4. Final Response Construction
        final_data = BillData(
            pagewise_line_items=llm_output_list,
            total_item_count=total_item_count,
        )
        
        return ExtractionResponse(
            is_success=True,
            token_usage=tracker.get_token_usage(),
            data=final_data
        )

    except APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API Error: {e}")
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"LLM output validation error. Model output did not match schema.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")