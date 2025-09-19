from fastapi import APIRouter, HTTPException, Request
from app.models.schemas import (
    ChatRequest, SummaryResponse, TimelineResponse,
    SentimentResponse, DashboardResponse, CompleteAnalysisResponse, ChatMessage
)
from typing import List, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
router = APIRouter()
# ---------- SUMMARY ----------
@router.post("/summary", response_model=SummaryResponse)
async def get_summary(request: ChatRequest, req: Request):
    """Generate chat summary with guaranteed bullet point format"""
    try:
        model_service = req.app.state.model_service
        summary = model_service.generate_summary(request.messages)

        # Ensure bullet format
        lines = summary.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not line.startswith('-'):
                line = f"- {line}"
            formatted_lines.append(line)

        final_summary = '\n'.join(formatted_lines) if formatted_lines else "- No summary available"
        return SummaryResponse(summary=final_summary)

    except Exception as e:
        print(f"Summary generation error: {e}")
        # Emergency fallback
        try:
            if request.messages:
                fallback_bullets = []
                for i, msg in enumerate(request.messages[:5]):
                    if msg.message:
                        user = (msg.user or "user").capitalize()
                        clean_msg = ' '.join(msg.message.split()[:12])
                        fallback_bullets.append(f"- {user}: {clean_msg}")
                fallback_summary = '\n'.join(fallback_bullets) if fallback_bullets else "- No messages found"
                return SummaryResponse(summary=fallback_summary)
            else:
                return SummaryResponse(summary="- No messages to summarize")
        except:
            return SummaryResponse(summary="- Error generating summary")


# ---------- TIMELINE ----------
@router.post("/timeline", response_model=TimelineResponse)
async def get_timeline(request: ChatRequest, req: Request):
    """Generate 24-hour timeline"""
    try:
        model_service = req.app.state.model_service
        timeline = model_service.create_timeline(request.messages)
        return TimelineResponse(timeline_24h=timeline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- SENTIMENT ----------
@router.post("/sentiment", response_model=SentimentResponse)
async def get_sentiment_analysis(request: ChatRequest, req: Request):
    """Analyze user sentiments"""
    try:
        model_service = req.app.state.model_service
        sentiment_analysis = model_service.analyze_sentiment(request.messages)
        return SentimentResponse(sentiment_analysis=sentiment_analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- DASHBOARD ----------
@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(req: Request, time_range: str = "24h"):
    """Get comprehensive dashboard analytics with visual data"""
    try:
        database = req.app.state.database
        collection = database.messages

        # Calculate time range
        now = datetime.utcnow()
        if time_range == "1h":
            start_time = now - timedelta(hours=1)
        elif time_range == "7d":
            start_time = now - timedelta(days=7)
        elif time_range == "30d":
            start_time = now - timedelta(days=30)
        else:  # default 24h
            start_time = now - timedelta(days=1)

        # Fetch messages
        cursor = collection.find({
            "timestamp": {"$gte": start_time.isoformat()}
        }).sort("timestamp", 1)

        messages_data = await cursor.to_list(length=5000)

        if not messages_data:
            return DashboardResponse(dashboard={
                "total_messages": 0,
                "active_users": 0,
                "user_message_counts": {},
                "hourly_message_counts": {},
                "sentiment_distribution": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "top_topics": [],
                "response_time_avg": 0.0,
                "issues_resolved": 0,
                "active_threads": 0,
                "peak_activity_hour": None,
                "most_active_user": None,
                "avg_message_length": 0.0,      # <-- added
                "total_words": 0,               # <-- added
                "conversation_threads": []      # <-- added
            })

        messages = [
            ChatMessage(
                user=msg.get("user", "unknown"),
                message=msg.get("message", ""),
                timestamp=msg.get("timestamp", ""),
                message_index=i
            ) for i, msg in enumerate(messages_data)
        ]

        model_service = req.app.state.model_service
        analytics = generate_visual_dashboard_data(messages, model_service, time_range)

        return DashboardResponse(dashboard=analytics)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- HELPERS ----------
def generate_visual_dashboard_data(messages: List[ChatMessage], model_service, time_range: str) -> Dict[str, Any]:
    """Generate rich analytics data for visual dashboard"""
    total_messages = len(messages)
    unique_users = set(msg.user for msg in messages if msg.user)
    active_users = len(unique_users)

    user_counts = Counter(msg.user for msg in messages if msg.user)

    time_counts = defaultdict(int)
    response_times = []

    for i, msg in enumerate(messages):
        try:
            dt = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))

            if time_range == "1h":
                minute_group = (dt.minute // 5) * 5
                time_key = dt.replace(minute=minute_group, second=0, microsecond=0).isoformat() + 'Z'
            elif time_range in ["7d", "30d"]:
                time_key = dt.strftime('%Y-%m-%dT00:00:00Z')
            else:
                time_key = dt.strftime('%Y-%m-%dT%H:00:00Z')

            time_counts[time_key] += 1

            if i > 0:
                prev_time = datetime.fromisoformat(messages[i - 1].timestamp.replace('Z', '+00:00'))
                time_diff = (dt - prev_time).total_seconds() / 60
                if time_diff < 60:
                    response_times.append(time_diff)

        except Exception:
            continue

    sentiment_results = model_service.analyze_sentiment(messages)
    sentiment_counts = Counter()

    for sentiment in sentiment_results.values():
        base_sentiment = sentiment.split('(')[0].strip()
        if 'positive' in base_sentiment.lower():
            sentiment_counts['Positive'] += 1
        elif 'negative' in base_sentiment.lower():
            sentiment_counts['Negative'] += 1
        else:
            sentiment_counts['Neutral'] += 1

    top_topics = model_service.extract_topics(messages)

    avg_response_time = statistics.mean(response_times) if response_times else 0.0

    issues_resolved = sum(1 for msg in messages
                          if any(word in msg.message.lower()
                                 for word in ['resolved', 'fixed', 'solved', 'closed', 'completed']))

    active_threads = estimate_active_threads(messages)

    return {
        "total_messages": total_messages,
        "active_users": active_users,
        "user_message_counts": dict(user_counts),
        "hourly_message_counts": dict(time_counts),
        "sentiment_distribution": {
            "Positive": sentiment_counts.get('Positive', 0),
            "Neutral": sentiment_counts.get('Neutral', 0),
            "Negative": sentiment_counts.get('Negative', 0)
        },
        "top_topics": top_topics,
        "response_time_avg": round(avg_response_time, 1),
        "issues_resolved": issues_resolved,
        "active_threads": active_threads,
        "peak_activity_hour": max(time_counts.items(), key=lambda x: x[1])[0] if time_counts else None,
        "most_active_user": max(user_counts.items(), key=lambda x: x[1])[0] if user_counts else None,
        "avg_message_length": sum(len(msg.message.split()) for msg in messages) / len(messages) if messages else 0,
        "total_words": sum(len(msg.message.split()) for msg in messages),
        "conversation_threads": estimate_conversation_threads(messages)
    }


def estimate_active_threads(messages: List[ChatMessage]) -> int:
    """Estimate number of active conversation threads"""
    if not messages:
        return 0

    threads = []
    current_thread = []

    for i, msg in enumerate(messages):
        if i == 0:
            current_thread = [msg]
            continue

        try:
            prev_time = datetime.fromisoformat(messages[i - 1].timestamp.replace('Z', '+00:00'))
            curr_time = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))
            time_gap = (curr_time - prev_time).total_seconds() / 60

            if time_gap > 30:
                if current_thread:
                    threads.append(current_thread)
                current_thread = [msg]
            else:
                current_thread.append(msg)

        except Exception:
            continue

    if current_thread:
        threads.append(current_thread)

    now = datetime.utcnow()
    active_count = 0

    for thread in threads:
        if thread:
            last_msg_time = datetime.fromisoformat(thread[-1].timestamp.replace('Z', '+00:00'))
            if (now - last_msg_time).total_seconds() / 3600 < 2:
                active_count += 1

    return max(1, active_count)


def estimate_conversation_threads(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Estimate conversation threads with topics and participants"""
    if not messages:
        return []

    threads = []
    current_thread = {"messages": [], "participants": set(), "start_time": None, "last_activity": None}

    for i, msg in enumerate(messages):
        if i == 0:
            current_thread = {
                "messages": [msg],
                "participants": {msg.user},
                "start_time": msg.timestamp,
                "last_activity": msg.timestamp
            }
            continue

        try:
            prev_time = datetime.fromisoformat(messages[i - 1].timestamp.replace('Z', '+00:00'))
            curr_time = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))
            time_gap = (curr_time - prev_time).total_seconds() / 60

            if time_gap > 30:
                if current_thread["messages"]:
                    thread_info = {
                        "id": len(threads) + 1,
                        "message_count": len(current_thread["messages"]),
                        "participant_count": len(current_thread["participants"]),
                        "participants": list(current_thread["participants"]),
                        "duration_minutes": calculate_thread_duration(current_thread["start_time"], current_thread["last_activity"]),
                        "topic": extract_thread_topic(current_thread["messages"])
                    }
                    threads.append(thread_info)

                current_thread = {
                    "messages": [msg],
                    "participants": {msg.user},
                    "start_time": msg.timestamp,
                    "last_activity": msg.timestamp
                }
            else:
                current_thread["messages"].append(msg)
                current_thread["participants"].add(msg.user)
                current_thread["last_activity"] = msg.timestamp

        except Exception:
            continue

    if current_thread["messages"]:
        thread_info = {
            "id": len(threads) + 1,
            "message_count": len(current_thread["messages"]),
            "participant_count": len(current_thread["participants"]),
            "participants": list(current_thread["participants"]),
            "duration_minutes": calculate_thread_duration(current_thread["start_time"], current_thread["last_activity"]),
            "topic": extract_thread_topic(current_thread["messages"])
        }
        threads.append(thread_info)

    return threads


def calculate_thread_duration(start_time: str, end_time: str) -> int:
    """Calculate thread duration in minutes"""
    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        return int((end - start).total_seconds() / 60)
    except:
        return 0


def extract_thread_topic(messages: List[ChatMessage]) -> str:
    """Extract main topic from a thread of messages"""
    if not messages:
        return "General Discussion"

    combined_text = " ".join(msg.message.lower() for msg in messages if msg.message)

    topic_keywords = {
        "Server Issues": ["server", "down", "api", "database", "error", "crash", "outage"],
        "Deployment": ["deploy", "release", "staging", "production", "build", "ci/cd"],
        "Bug Reports": ["bug", "issue", "problem", "broken", "fix", "error"],
        "Code Review": ["review", "code", "merge", "pull request", "pr", "commit"],
        "Meeting": ["meeting", "call", "schedule", "agenda", "discuss"],
        "Client Work": ["client", "demo", "presentation", "deadline", "requirement"],
        "Team Updates": ["update", "status", "progress", "team", "standup"]
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in combined_text for keyword in keywords):
            return topic

    return "General Discussion"


# ---------- COMPLETE ----------
# Replace the /complete route in your chat_routes.py with this:

# Replace your /complete route with this debugging version temporarily:

@router.post("/complete", response_model=CompleteAnalysisResponse)
async def get_complete(request: ChatRequest, req: Request):
    """Run full analysis in one call"""
    try:
        model_service = req.app.state.model_service
        
        # Generate all analyses
        summary = model_service.generate_summary(request.messages)
        timeline = model_service.create_timeline(request.messages)
        sentiment_analysis = model_service.analyze_sentiment(request.messages)
        
        # Debug the dashboard stats
        dashboard_stats = model_service.generate_dashboard_stats(request.messages)
        
        print("=== DEBUG: Dashboard Stats Output ===")
        print(f"Type: {type(dashboard_stats)}")
        print(f"Keys: {list(dashboard_stats.keys()) if isinstance(dashboard_stats, dict) else 'Not a dict'}")
        print(f"Content: {dashboard_stats}")
        print("=====================================")
        
        # Check if we have all required fields
        required_fields = [
            'total_messages', 'active_users', 'user_message_counts', 
            'hourly_message_counts', 'sentiment_distribution', 'top_topics',
            'response_time_avg', 'issues_resolved', 'active_threads',
            'avg_message_length', 'total_words', 'conversation_threads'
        ]
        
        missing_fields = [field for field in required_fields if field not in dashboard_stats]
        if missing_fields:
            print(f"MISSING FIELDS: {missing_fields}")
            
            # Add missing fields with default values
            for field in missing_fields:
                if field == 'active_users':
                    dashboard_stats[field] = len(set(msg.user for msg in request.messages))
                elif field == 'sentiment_distribution':
                    dashboard_stats[field] = {"Positive": 0, "Neutral": 0, "Negative": 0}
                elif field == 'response_time_avg':
                    dashboard_stats[field] = 0.0
                elif field == 'issues_resolved':
                    dashboard_stats[field] = 0
                elif field == 'active_threads':
                    dashboard_stats[field] = 1
                elif field == 'avg_message_length':
                    dashboard_stats[field] = 0.0
                elif field == 'total_words':
                    dashboard_stats[field] = 0
                elif field == 'conversation_threads':
                    dashboard_stats[field] = []
                elif field == 'peak_activity_hour':
                    dashboard_stats[field] = None
                elif field == 'most_active_user':
                    dashboard_stats[field] = None

        return CompleteAnalysisResponse(
            summary=summary,
            timeline_24h=timeline,
            sentiment_analysis=sentiment_analysis,
            dashboard=dashboard_stats
        )
        
    except Exception as e:
        import traceback
        print(f"Complete analysis error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    
