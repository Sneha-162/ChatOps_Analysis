
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime

class ChatMessage(BaseModel):
    user: str
    message: str
    timestamp: str
    message_index: Optional[int] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class SummaryResponse(BaseModel):
    summary: str

class TimelineEvent(BaseModel):
    timestamp: str
    user: str
    message_index: int
    topic: str

class TimelineResponse(BaseModel):
    timeline_24h: List[TimelineEvent]

class SentimentResponse(BaseModel):
    sentiment_analysis: Dict[str, str]

class EnhancedDashboardStats(BaseModel):
    total_messages: int
    active_users: int
    user_message_counts: Dict[str, int]
    hourly_message_counts: Dict[str, int]
    sentiment_distribution: Dict[str, int]
    top_topics: List[str]
    response_time_avg: float
    issues_resolved: int
    active_threads: int
    peak_activity_hour: Optional[str] = None
    most_active_user: Optional[str] = None
    avg_message_length: float
    total_words: int
    conversation_threads: List[Dict[str, Any]]

class EnhancedDashboardResponse(BaseModel):
    dashboard: EnhancedDashboardStats

class DashboardResponse(BaseModel):
    dashboard: EnhancedDashboardStats

class CompleteAnalysisResponse(BaseModel):
    summary: str
    timeline_24h: List[TimelineEvent]
    sentiment_analysis: Dict[str, str]
    dashboard: EnhancedDashboardStats
