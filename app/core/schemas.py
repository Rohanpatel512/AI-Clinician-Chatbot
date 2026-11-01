from pydantic import BaseModel, Field
from typing import List, Literal, Optional 


class QueryProcessor(BaseModel):
    rewritten_query: str = Field(description="Rewritten Standalone Query")
    intent: Literal["GENERAL", "ACTIVE_CONCERN"] = Field(description="GENERAL if purely education, ACTIVE_CONCERN if someone is actively experiencing symptoms.")
    urgency: Literal["CRITICAL", "ACUTE", "SUBACUTE", "NONE"] = Field(description="Level of urgency based of users query and conversation history")
