from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class ACLPerformative(str, Enum):
    PROPOSE = "PROPOSE"
    CHALLENGE = "CHALLENGE"
    VERIFY = "VERIFY"
    CONFIRM = "CONFIRM"

class ACLMessage(BaseModel):
    performative: ACLPerformative
    sender: str
    receiver: str
    content: str
    reply_with: str
    in_reply_to: Optional[str] = None
    language: str = "ACL"
    ontology: str = "MarketPrediction"
    protocol: str = "Debate"
    conversation_id: str = "debate1"

class InputSchema(BaseModel):
    conversation: List[ACLMessage] = Field(..., title="Conversation")
    agent_name: str = Field(..., title="Agent name")
    agent_type: str = Field(..., title="Agent type")