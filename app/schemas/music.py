from pydantic import BaseModel

class RecommendRequest(BaseModel):
    track_id: str
    n: int = 10