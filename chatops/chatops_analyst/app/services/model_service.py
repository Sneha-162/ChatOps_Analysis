# Complete ModelService - Replace your entire model_service.py with this:

import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from collections import Counter, defaultdict
from datetime import datetime
import re
import statistics
from typing import List, Dict, Any
from app.models.schemas import ChatMessage, TimelineEvent


class ModelService:
    def __init__(self):  # Fixed constructor name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summarizer = None
        self.sentiment_analyzer = None
        self.embedding_model = None

    async def initialize(self):
        """Initialize all models"""
        print(f"Initializing models on device: {self.device}")
        device = 0 if torch.cuda.is_available() else -1

        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device,
        )

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        if self.device.type != "cpu":
            self.embedding_model = self.embedding_model.to(self.device)

        print("Models initialized successfully!")

    def generate_summary(self, messages: List[ChatMessage]) -> str:
        """Generate a concise bullet-point summary of the chat (5 short bullets)."""
        return self.create_manual_bullets(messages, max_points=5)

    def create_manual_bullets(self, messages: List[ChatMessage], max_points: int = 5) -> str:
        """Create structured bullet points from chat messages"""
        def truncate_words(s: str, n: int) -> str:
            w = s.split()
            return " ".join(w[:n])

        if not messages:
            return "- No messages to summarize"

        bullets = []
        processed_events = set()

        for i, msg in enumerate(messages):
            if not msg.message or not msg.message.strip():
                continue

            message = msg.message.strip()
            message_lower = message.lower()
            user = (msg.user or "user").capitalize()

            if len(message.split()) < 3:
                continue

            bullet_text = ""

            if re.search(r'\b(shipped|deployed|released|published)\b', message_lower):
                if 'staging' in message_lower:
                    bullet_text = f"{user} shipped/deployed to staging environment"
                elif 'production' in message_lower or 'prod' in message_lower:
                    bullet_text = f"{user} deployed to production"
                else:
                    bullet_text = f"{user} completed deployment/release"
            elif re.search(r'\b(bug|issue|problem|error|broken)\b', message_lower):
                if re.search(r'\b(found|discovered|identified|noticed)\b', message_lower):
                    if 'login' in message_lower:
                        bullet_text = f"{user} discovered a login-related issue"
                    else:
                        bullet_text = f"{user} found a system issue requiring attention"
                elif re.search(r'\b(fixed|resolved|solved|corrected)\b', message_lower):
                    bullet_text = f"{user} resolved the reported issue"
            elif re.search(r'\b(test|testing|check|checking|verify|verifying)\b', message_lower):
                if re.search(r'\b(passed|successful|working|good)\b', message_lower):
                    bullet_text = f"{user} confirmed systems are working properly"
                else:
                    bullet_text = f"{user} is performing system testing"
            elif re.search(r'\b(confirmed|verified|completed|finished|done)\b', message_lower):
                if 'login' in message_lower:
                    bullet_text = f"{user} confirmed login functionality is working"
                else:
                    bullet_text = f"{user} provided positive status update"
            elif re.search(r'\b(frustrated|relieved|worried|concerned|happy)\b', message_lower):
                if 'frustrated' in message_lower:
                    bullet_text = f"{user} expressed frustration with system issues"
                elif 'relieved' in message_lower:
                    bullet_text = f"{user} expressed relief that issues were resolved"

            if not bullet_text:
                clean_msg = re.sub(r'\b(the|a|an|and|or|but|so|then|now|just|really|very)\b', ' ', message_lower)
                clean_msg = ' '.join(clean_msg.split())

                words = message.split()
                if len(words) > 8:
                    key_part = ' '.join(words[:8]) + "..."
                else:
                    key_part = message

                bullet_text = f"{user}: {key_part}"

            bullet_key = bullet_text.lower().replace(user.lower(), "").strip()
            if bullet_key not in processed_events and bullet_text:
                bullets.append(f"- {bullet_text}")
                processed_events.add(bullet_key)

                if len(bullets) >= max_points:
                    break

        while len(bullets) < min(max_points, len(messages)) and len(bullets) < len(messages):
            for msg in messages[len(bullets):]:
                if msg.message and msg.message.strip():
                    user = (msg.user or "user").capitalize()
                    clean_msg = truncate_words(msg.message.strip(), 12)
                    simple_bullet = f"- {user}: {clean_msg}"
                    if simple_bullet not in bullets:
                        bullets.append(simple_bullet)
                        break

        return "\n".join(bullets) if bullets else "- No significant events to summarize"

    def analyze_sentiment(self, messages: List[ChatMessage]) -> Dict[str, str]:
        """Analyze sentiment per user"""
        user_messages = defaultdict(list)
        for msg in messages:
            if msg.user and msg.message:
                user_messages[msg.user].append(msg.message)

        user_sentiments = {}
        for user, msgs in user_messages.items():
            combined_text = " ".join(msgs)
            try:
                result = self.sentiment_analyzer(combined_text[:512])
                sentiment_label = result[0]["label"]
                confidence = result[0]["score"]

                if "POSITIVE" in sentiment_label.upper() or "POS" in sentiment_label.upper():
                    sentiment = "Positive"
                elif "NEGATIVE" in sentiment_label.upper() or "NEG" in sentiment_label.upper():
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                if confidence > 0.8:
                    user_sentiments[user] = f"{sentiment} (confident)"
                else:
                    user_sentiments[user] = sentiment
            except:
                blob = TextBlob(combined_text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    user_sentiments[user] = "Positive"
                elif polarity < -0.1:
                    user_sentiments[user] = "Negative"
                else:
                    user_sentiments[user] = "Neutral"
        return user_sentiments

    def extract_topics(self, messages: List[ChatMessage]) -> List[str]:
        """Extract topics using keyword extraction"""
        if not messages:
            return []
            
        texts = [msg.message for msg in messages if msg.message]
        all_text = " ".join(texts).lower()
        
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must", "can", "i", "you", "he",
            "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
            "her", "its", "our", "their", "now", "then", "just"
        }

        words = re.findall(r"\b\w+\b", all_text)
        filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]

        word_counts = Counter(filtered_words)
        topics = [word.capitalize() for word, _ in word_counts.most_common(5)]

        domain_topics = []
        if any(word in all_text for word in ["server", "api", "database", "down", "error"]):
            domain_topics.append("Server issues")
        if any(word in all_text for word in ["demo", "presentation", "slides", "client"]):
            domain_topics.append("Client demo")
        if any(word in all_text for word in ["meeting", "schedule", "time", "tomorrow"]):
            domain_topics.append("Scheduling")
        if any(word in all_text for word in ["monitoring", "alerts", "system"]):
            domain_topics.append("System monitoring")
        if any(word in all_text for word in ["bug", "issue", "problem", "fix"]):
            domain_topics.append("Bug fixing")
        if any(word in all_text for word in ["deploy", "staging", "production"]):
            domain_topics.append("Deployment")

        all_topics = domain_topics + topics
        return list(dict.fromkeys(all_topics))[:5]  # Remove duplicates, keep first 5

    def create_timeline(self, messages: List[ChatMessage]) -> List[TimelineEvent]:
        """Create timeline with topic tagging"""
        timeline = []

        for i, msg in enumerate(messages):
            if not msg.message:
                continue
                
            message_lower = msg.message.lower()
            assigned_topic = "General discussion"

            if any(word in message_lower for word in ["server", "api", "database", "down", "error"]):
                assigned_topic = "Server issues"
            elif any(word in message_lower for word in ["demo", "presentation", "slides", "client"]):
                assigned_topic = "Client demo"
            elif any(word in message_lower for word in ["meeting", "schedule", "time", "tomorrow"]):
                assigned_topic = "Scheduling"
            elif any(word in message_lower for word in ["monitoring", "alerts", "system"]):
                assigned_topic = "System monitoring"
            elif any(word in message_lower for word in ["frustrated", "relieved", "happy", "worried"]):
                assigned_topic = "Team morale"
            elif any(word in message_lower for word in ["bug", "issue", "problem", "fix"]):
                assigned_topic = "Bug fixing"
            elif any(word in message_lower for word in ["deploy", "staging", "production"]):
                assigned_topic = "Deployment"

            timeline.append(
                TimelineEvent(
                    timestamp=msg.timestamp,
                    user=msg.user,
                    message_index=i,
                    topic=assigned_topic,
                )
            )

        return timeline

    def generate_dashboard_stats(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Generate comprehensive dashboard statistics matching EnhancedDashboardStats schema"""
        if not messages:
            return {
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
                "avg_message_length": 0.0,
                "total_words": 0,
                "conversation_threads": []
            }
        
        total_messages = len(messages)
        
        # User analysis
        user_counts = Counter(msg.user for msg in messages if msg.user)
        unique_users = set(msg.user for msg in messages if msg.user)
        active_users = len(unique_users)
        
        # Time-based analysis
        hourly_counts = defaultdict(int)
        response_times = []
        
        for i, msg in enumerate(messages):
            try:
                dt = datetime.fromisoformat(msg.timestamp.replace("Z", "+00:00"))
                hour_key = dt.strftime("%Y-%m-%dT%H:00:00Z")
                hourly_counts[hour_key] += 1
                
                # Calculate response times between consecutive messages
                if i > 0:
                    prev_time = datetime.fromisoformat(messages[i-1].timestamp.replace('Z', '+00:00'))
                    time_diff = (dt - prev_time).total_seconds() / 60  # minutes
                    if time_diff < 60:  # Only count if within an hour
                        response_times.append(time_diff)
            except:
                continue

        # Word analysis
        total_words = 0
        message_lengths = []
        for msg in messages:
            if msg.message:
                word_count = len(msg.message.split())
                total_words += word_count
                message_lengths.append(word_count)
        
        avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0.0
        
        # Sentiment analysis
        sentiment_results = self.analyze_sentiment(messages)
        sentiment_counts = Counter()
        
        for sentiment in sentiment_results.values():
            # Parse sentiment (handle formats like "Positive (confident)")
            base_sentiment = sentiment.split('(')[0].strip()
            if 'positive' in base_sentiment.lower():
                sentiment_counts['Positive'] += 1
            elif 'negative' in base_sentiment.lower():
                sentiment_counts['Negative'] += 1
            else:
                sentiment_counts['Neutral'] += 1
        
        # Topics
        top_topics = self.extract_topics(messages)
        
        # Advanced metrics
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        
        # Count issues resolved (simple heuristic)
        issues_resolved = sum(1 for msg in messages 
                             if any(word in msg.message.lower() 
                                   for word in ['resolved', 'fixed', 'solved', 'closed', 'completed']))
        
        # Estimate active threads
        active_threads = self.estimate_active_threads(messages)
        
        # Find peak activity hour
        peak_activity_hour = max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None
        
        # Find most active user
        most_active_user = max(user_counts.items(), key=lambda x: x[1])[0] if user_counts else None
        
        # Generate conversation threads
        conversation_threads = self.estimate_conversation_threads(messages)
        
        return {
            "total_messages": total_messages,
            "active_users": active_users,
            "user_message_counts": dict(user_counts),
            "hourly_message_counts": dict(hourly_counts),
            "sentiment_distribution": {
                "Positive": sentiment_counts.get('Positive', 0),
                "Neutral": sentiment_counts.get('Neutral', 0),
                "Negative": sentiment_counts.get('Negative', 0)
            },
            "top_topics": top_topics,
            "response_time_avg": round(avg_response_time, 1),
            "issues_resolved": issues_resolved,
            "active_threads": active_threads,
            "peak_activity_hour": peak_activity_hour,
            "most_active_user": most_active_user,
            "avg_message_length": round(avg_message_length, 1),
            "total_words": total_words,
            "conversation_threads": conversation_threads
        }

    def estimate_active_threads(self, messages: List[ChatMessage]) -> int:
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
                prev_time = datetime.fromisoformat(messages[i-1].timestamp.replace('Z', '+00:00'))
                curr_time = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))
                time_gap = (curr_time - prev_time).total_seconds() / 60  # minutes
                
                # If there's a gap of more than 30 minutes, consider it a new thread
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
        
        # Consider threads with recent activity (last 2 hours) as active
        now = datetime.utcnow()
        active_count = 0
        
        for thread in threads:
            if thread:
                try:
                    last_msg_time = datetime.fromisoformat(thread[-1].timestamp.replace('Z', '+00:00'))
                    if (now - last_msg_time).total_seconds() / 3600 < 2:  # within 2 hours
                        active_count += 1
                except:
                    continue
        
        return max(1, active_count)  # At least 1 thread if there are messages

    def estimate_conversation_threads(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
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
                prev_time = datetime.fromisoformat(messages[i-1].timestamp.replace('Z', '+00:00'))
                curr_time = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))
                time_gap = (curr_time - prev_time).total_seconds() / 60
                
                if time_gap > 30:  # New thread
                    if current_thread["messages"]:
                        # Finalize current thread
                        thread_info = {
                            "id": len(threads) + 1,
                            "message_count": len(current_thread["messages"]),
                            "participant_count": len(current_thread["participants"]),
                            "participants": list(current_thread["participants"]),
                            "duration_minutes": self.calculate_thread_duration(current_thread["start_time"], current_thread["last_activity"]),
                            "topic": self.extract_thread_topic(current_thread["messages"])
                        }
                        threads.append(thread_info)
                    
                    # Start new thread
                    current_thread = {
                        "messages": [msg],
                        "participants": {msg.user},
                        "start_time": msg.timestamp,
                        "last_activity": msg.timestamp
                    }
                else:  # Continue current thread
                    current_thread["messages"].append(msg)
                    current_thread["participants"].add(msg.user)
                    current_thread["last_activity"] = msg.timestamp
                    
            except Exception:
                continue
        
        # Don't forget the last thread
        if current_thread["messages"]:
            thread_info = {
                "id": len(threads) + 1,
                "message_count": len(current_thread["messages"]),
                "participant_count": len(current_thread["participants"]),
                "participants": list(current_thread["participants"]),
                "duration_minutes": self.calculate_thread_duration(current_thread["start_time"], current_thread["last_activity"]),
                "topic": self.extract_thread_topic(current_thread["messages"])
            }
            threads.append(thread_info)
        
        return threads

    def calculate_thread_duration(self, start_time: str, end_time: str) -> int:
        """Calculate thread duration in minutes"""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return int((end - start).total_seconds() / 60)
        except:
            return 0

    def extract_thread_topic(self, messages: List[ChatMessage]) -> str:
        """Extract main topic from a thread of messages"""
        if not messages:
            return "General Discussion"
        
        # Combine all messages in the thread
        combined_text = " ".join(msg.message.lower() for msg in messages if msg.message)
        
        # Simple keyword-based topic detection
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