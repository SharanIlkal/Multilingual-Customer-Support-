from dataclasses import dataclass
from typing import Optional


@dataclass
class CustomerSupportRecord:
    """
    Standard representation of one customer-support message.
    """

    text: str
    language: str
    intent: str
    sentiment: str
    priority: str
    category: str
    source: Optional[str] = None