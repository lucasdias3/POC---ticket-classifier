from pydantic import BaseModel
from typing import List, Optional

class Ticket(BaseModel):
    ticket_id: int
    customer_segment: str
    channel: str
    text: str
    category: str
    priority: str

class TicketRequest(BaseModel):
    customer_segment: str
    channel: str
    text: str

class TicketResponse(BaseModel):
    ticket_id: int
    category: str
    priority: str

class ClassificationResult(BaseModel):
    tickets: List[TicketResponse]
    message: Optional[str] = None