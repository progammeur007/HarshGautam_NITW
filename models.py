# models.py

from pydantic import BaseModel, Field, RootModel
from typing import List, Literal, Optional, Any, Union # Added Any for robust parsing

# --- 1. Sub-Schemas for Line Items and Tokens ---

class BillItem(BaseModel):
    """Schema for a single line item, matching the required fields."""
    item_name: str = Field(..., description="Name of the item exactly as mentioned in the bill.")
    item_amount: float = Field(..., description="Net Amount of the item post discounts as mentioned in the bill, must be a float.")
    item_rate: float = Field(..., description="Rate per unit exactly as mentioned in the bill, must be a float.")
    item_quantity: float = Field(..., description="Quantity exactly as mentioned in the bill, must be a float.")

class TokenUsage(BaseModel):
    """Schema for tracking cumulative token usage, all integers."""
    total_tokens: int = Field(..., description="Cumulative Tokens from all LLM calls (Input + Output).")
    input_tokens: int = Field(..., description="Cumulative Input/Prompt Tokens from all LLM calls.")
    output_tokens: int = Field(..., description="Cumulative Output/Candidate Tokens from all LLM calls.")

# --- 2. Nested Pagemise Data Schema ---

# Define the allowed page types for classification
PageType = Literal["Bill Detail", "Final Bill", "Pharmacy", "Other"]

class PagewiseLineItems(BaseModel):
    """Container for line items on a single page, including page classification."""
    page_no: str = Field(..., description="The page number from which the items were extracted (e.g., '1', '2').")
    page_type: PageType = Field(..., description="Classification of the page content: Bill Detail, Final Bill, Pharmacy, or Other.")
    bill_items: List[BillItem] = Field(..., description="List of all extracted line items for this page.")

# --- 3. RootModel Fix for Gemini API ---

class PagewiseListRoot(RootModel):
    """
    CRITICAL FIX: Wraps the list of pages to simplify the JSON schema 
    for the Gemini API's structured output.
    """
    root: List[PagewiseLineItems]

# --- 4. KVP Extraction Schema (Internal Use for Accuracy Check) ---

class FinalTotalKVP(BaseModel):
    """Temporary schema to extract the single 'Actual Bill Total' and sub-total for validation."""
    actual_bill_total: float = Field(..., description="The final total amount (e.g., Grand Total, Net Payable Amount) as explicitly printed on the bill, must be a float.")
    sub_total: Optional[float] = Field(None, description="The sub-total amount, if explicitly printed on the bill, otherwise null.")
    
# --- 5. Final Data and Request Schemas ---

class BillData(BaseModel):
    """The core 'data' object in the response (matches submission spec)."""
    pagewise_line_items: List[PagewiseLineItems]
    total_item_count: int = Field(..., description="Total count of unique bill_items across all pages.")

class ExtractionRequest(BaseModel):
    """The Request Body schema."""
    document: str = Field(..., description="URL of the document (image/PDF) to be processed.")

class ExtractionResponse(BaseModel):
    """The Final API Response schema, matching the submission format."""
    is_success: bool
    token_usage: TokenUsage
    data: BillData