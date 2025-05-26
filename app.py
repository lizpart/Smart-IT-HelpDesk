import os
import time
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import openai
import logging
from datetime import datetime, timedelta
import uvicorn
import requests
import base64
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams
import pyngrok.conf
from pyngrok import ngrok
import uuid
from email_service import EmailService
import sqlite3
from threading import Lock

# Load environment variables
load_dotenv()

# Configure logging with better formatting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diagnostic_assistant.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = "+14155238886"  # Twilio sandbox number
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    NGROK_AUTH_TOKEN: str = os.getenv("NGROK_AUTH_TOKEN", "")
    SENDGRID_API_KEY: str = os.getenv("SENDGRID_API_KEY", "")
    NOTIFICATION_EMAIL: str = os.getenv("NOTIFICATION_EMAIL", "")
    FROM_EMAIL: str = os.getenv("FROM_EMAIL", "")
    USER_EMAILS: dict = json.loads(os.getenv("USER_EMAILS", "{}"))
    
    # Memory settings
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))  # Number of messages to remember
    CONVERSATION_TIMEOUT_HOURS: int = int(os.getenv("CONVERSATION_TIMEOUT_HOURS", "24"))  # Hours after which conversation expires
    
    class Config:
        env_file = ".env"
        extra = "allow" 

settings = Settings()

# Configure ngrok with better error handling
try:
    pyngrok.conf.get_default().auth_token = settings.NGROK_AUTH_TOKEN
except Exception as e:
    logger.warning(f"Could not set ngrok auth token: {e}")

# Initialize FastAPI app
app = FastAPI()

class ConversationMemory:
    """Handles conversation history storage and retrieval"""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self.lock = Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for conversation storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        message_type TEXT NOT NULL,  -- 'user' or 'assistant'
                        content TEXT NOT NULL,
                        media_url TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        session_id TEXT NOT NULL
                    )
                ''')
                
                # Create index for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_user_timestamp 
                    ON conversations(user_id, timestamp DESC)
                ''')
                
                conn.commit()
                logger.info("Conversation database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing conversation database: {e}")
            raise
    
    def _get_session_id(self, user_id: str) -> str:
        """Get or create session ID for user based on recent activity"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check for recent conversation (within timeout period)
                timeout_timestamp = datetime.now() - timedelta(hours=settings.CONVERSATION_TIMEOUT_HOURS)
                cursor.execute('''
                    SELECT session_id FROM conversations 
                    WHERE user_id = ? AND timestamp > ? 
                    ORDER BY timestamp DESC LIMIT 1
                ''', (user_id, timeout_timestamp))
                
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:
                    # Create new session
                    return str(uuid.uuid4())
        except Exception as e:
            logger.error(f"Error getting session ID: {e}")
            return str(uuid.uuid4())
    
    def add_message(self, user_id: str, message_type: str, content: str, media_url: Optional[str] = None):
        """Add a message to conversation history"""
        try:
            with self.lock:
                session_id = self._get_session_id(user_id)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO conversations (user_id, message_type, content, media_url, session_id)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user_id, message_type, content, media_url, session_id))
                    conn.commit()
                    
                    # Clean up old messages to maintain conversation limit
                    self._cleanup_old_messages(cursor, user_id, session_id)
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error adding message to conversation history: {e}")
    
    def _cleanup_old_messages(self, cursor, user_id: str, session_id: str):
        """Remove old messages beyond the conversation limit"""
        try:
            # Get message count for current session
            cursor.execute('''
                SELECT COUNT(*) FROM conversations 
                WHERE user_id = ? AND session_id = ?
            ''', (user_id, session_id))
            
            count = cursor.fetchone()[0]
            
            if count > settings.MAX_CONVERSATION_HISTORY:
                # Remove oldest messages beyond limit
                excess_count = count - settings.MAX_CONVERSATION_HISTORY
                cursor.execute('''
                    DELETE FROM conversations 
                    WHERE user_id = ? AND session_id = ?
                    AND id IN (
                        SELECT id FROM conversations 
                        WHERE user_id = ? AND session_id = ?
                        ORDER BY timestamp ASC 
                        LIMIT ?
                    )
                ''', (user_id, session_id, user_id, session_id, excess_count))
                
        except Exception as e:
            logger.error(f"Error cleaning up old messages: {e}")
    
    def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Get recent conversation history for user"""
        try:
            session_id = self._get_session_id(user_id)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_type, content, media_url, timestamp 
                    FROM conversations 
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                ''', (user_id, session_id, settings.MAX_CONVERSATION_HISTORY))
                
                results = cursor.fetchall()
                
                history = []
                for row in results:
                    message_type, content, media_url, timestamp = row
                    history.append({
                        'role': message_type,
                        'content': content,
                        'media_url': media_url,
                        'timestamp': timestamp
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def clear_user_history(self, user_id: str):
        """Clear conversation history for a specific user"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM conversations WHERE user_id = ?', (user_id,))
                    conn.commit()
                    logger.info(f"Cleared conversation history for user: {user_id}")
        except Exception as e:
            logger.error(f"Error clearing user history: {e}")
    
    def cleanup_expired_conversations(self):
        """Remove conversations older than timeout period"""
        try:
            with self.lock:
                timeout_timestamp = datetime.now() - timedelta(hours=settings.CONVERSATION_TIMEOUT_HOURS * 2)  # Double timeout for cleanup
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM conversations WHERE timestamp < ?', (timeout_timestamp,))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} expired conversation messages")
        except Exception as e:
            logger.error(f"Error cleaning up expired conversations: {e}")

class KnowledgeBase:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small"  # Using latest embedding model
        )
        self.collection_name = "technical_docs"
        self.vector_size = 1536
        
        # Initialize Qdrant client with better error handling
        try:
            self.qdrant_client = QdrantClient(path="./qdrant_storage")
            self.initialize_knowledge_base()
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            # Fallback to in-memory client
            self.qdrant_client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant client as fallback")
            self.initialize_knowledge_base()

    def initialize_knowledge_base(self):
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)

            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                self._create_knowledge_base()
                logger.info("Created new knowledge base")
            else:
                logger.info("Loaded existing knowledge base")
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            # Create a minimal knowledge base for testing
            self._create_minimal_knowledge_base()

    def _create_knowledge_base(self):
        try:
            # Check if technical_docs directory exists
            if not os.path.exists("technical_docs/"):
                logger.warning("technical_docs directory not found, creating minimal knowledge base")
                self._create_minimal_knowledge_base()
                return

            loader = DirectoryLoader("technical_docs/", glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            
            if not documents:
                logger.warning("No documents found in technical_docs/, creating minimal knowledge base")
                self._create_minimal_knowledge_base()
                return

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            for i, doc in enumerate(texts):
                try:
                    embedding = self.embeddings.embed_query(doc.page_content)
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[models.PointStruct(
                            id=i, 
                            vector=embedding, 
                            payload={"text": doc.page_content}
                        )]
                    )
                except Exception as e:
                    logger.error(f"Error processing document {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error creating knowledge base: {str(e)}")
            self._create_minimal_knowledge_base()

    def _create_minimal_knowledge_base(self):
        """Create a minimal knowledge base for testing purposes"""
        try:
            sample_docs = [
                "For network connectivity issues, check cable connections and restart the router.",
                "If equipment is overheating, ensure proper ventilation and clean air filters.",
                "For software errors, try restarting the application or checking for updates.",
                "Power supply issues often require checking fuses and voltage levels.",
                "Regular maintenance includes cleaning, lubrication, and calibration checks."
            ]
            
            for i, doc_text in enumerate(sample_docs):
                try:
                    embedding = self.embeddings.embed_query(doc_text)
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[models.PointStruct(
                            id=i, 
                            vector=embedding, 
                            payload={"text": doc_text}
                        )]
                    )
                except Exception as e:
                    logger.error(f"Error creating minimal knowledge base entry {i}: {e}")
                    
            logger.info("Created minimal knowledge base for testing")
        except Exception as e:
            logger.error(f"Error creating minimal knowledge base: {e}")

    def get_relevant_context(self, query, k=3):
        try:
            query_embedding = self.embeddings.embed_query(query)
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k
            )
            context = "\n".join([hit.payload["text"] for hit in search_result])
            return context
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return "Basic troubleshooting: Check connections, restart system, and verify power supply."

class DiagnosticAssistant:
    def __init__(self):
        self.twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)  # Updated client initialization
        self.knowledge_base = KnowledgeBase()
        self.conversation_memory = ConversationMemory()  # Add conversation memory
        self.ngrok_tunnel = None
        
        # Initialize email service with error handling
        try:
            self.email_service = EmailService(
                settings.SENDGRID_API_KEY, 
                settings.FROM_EMAIL, 
                settings.NOTIFICATION_EMAIL
            )
        except Exception as e:
            logger.error(f"Failed to initialize email service: {e}")
            self.email_service = None
            
        self.user_emails = settings.USER_EMAILS

    def get_user_email(self, whatsapp_number: str) -> Optional[str]:
        """Get user email from WhatsApp number"""
        # Remove 'whatsapp:' prefix if present
        clean_number = whatsapp_number.replace('whatsapp:', '')
        return self.user_emails.get(clean_number)

    def _build_conversation_context(self, user_id: str, current_message: str) -> List[Dict]:
        """Build conversation context including history and current message"""
        messages = [
            {
                "role": "system",
                "content": """You are a technical support assistant for industrial/technical equipment. 
                You have access to conversation history and should reference previous messages when relevant.
                Provide clear, concise solutions based on the provided context and conversation history.
                If a user refers to a previous issue or asks follow-up questions, use the conversation history to provide contextual responses."""
            }
        ]
        
        # Get conversation history
        history = self.conversation_memory.get_conversation_history(user_id)
        
        # Add conversation history to messages
        for msg in history:
            if msg['role'] == 'user':
                content = msg['content']
                if msg['media_url']:
                    content += " [User sent an image]"
                messages.append({"role": "user", "content": content})
            elif msg['role'] == 'assistant':
                messages.append({"role": "assistant", "content": msg['content']})
        
        # Add current message
        messages.append({"role": "user", "content": current_message})
        
        return messages

    async def send_whatsapp_message(self, to: str, message: str):
        """Send WhatsApp message using Twilio"""
        try:
            from_whatsapp = f"whatsapp:{settings.TWILIO_PHONE_NUMBER}"
            to_whatsapp = to if to.startswith("whatsapp:") else f"whatsapp:{to}"
            
            logger.info(f"Sending message from {from_whatsapp} to {to_whatsapp}")
            
            message_chunks = self.chunk_message(message)
            
            for chunk in message_chunks:
                self.twilio_client.messages.create(
                    body=chunk,
                    from_=from_whatsapp,
                    to=to_whatsapp
                )
                if len(message_chunks) > 1:
                    time.sleep(1)
                    
            logger.info("Message sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
            raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")

    def chunk_message(self, message: str, max_length: int = 1500) -> list:
        if len(message) <= max_length:
            return [message]
        
        chunks = []
        current_chunk = ""
        sentences = message.replace("\n", ". ").split(". ")
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        if len(chunks) > 1:
            chunks = [f"({i+1}/{len(chunks)}) {chunk}" for i, chunk in enumerate(chunks)]
        
        return chunks

    async def check_hardware_issue(self, text: str) -> bool:
        """Check if the issue requires hardware maintenance"""
        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using more cost-effective model for classification
                messages=[
                    {
                        "role": "system",
                        "content": """You are a technical support assistant for industrial/technical equipment.
                        Only respond with 'true' if the issue requires hardware maintenance or physical intervention
                        on technical equipment. Respond with 'false' for:
                        - Non-technical queries
                        - Personal issues
                        - Medical issues
                        - Questions about pets or animals
                        - General inquiries
                        Only respond with 'true' or 'false'."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=10,
                temperature=0  # For consistent classification
            )
            response = completion.choices[0].message.content.lower().strip()
            return response == "true"
            
        except Exception as e:
            logger.error(f"Error checking hardware issue: {str(e)}")
            return False

    async def process_image(self, image_url: str, sender: str) -> str:
        try:
            response = requests.get(
                image_url,
                auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
                timeout=30
            )
            response.raise_for_status()
            
            image_data = base64.b64encode(response.content).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{image_data}"
            
            # Get conversation history for context
            history = self.conversation_memory.get_conversation_history(sender)
            
            # Build messages with conversation context
            messages = [
                {
                    "role": "system",
                    "content": "You are a technical support assistant. Analyze images and provide solutions based on conversation history when relevant."
                }
            ]
            
            # Add relevant conversation history
            for msg in history[-3:]:  # Last 3 messages for context
                if msg['role'] == 'user':
                    content = msg['content']
                    if msg['media_url']:
                        content += " [Previous image sent]"
                    messages.append({"role": "user", "content": content})
                elif msg['role'] == 'assistant':
                    messages.append({"role": "assistant", "content": msg['content']})
            
            # Add current image analysis request
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this technical issue or error message, considering our previous conversation:"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            })
            
            vision_completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Updated model name
                messages=messages,
                max_tokens=300
            )
            
            image_description = vision_completion.choices[0].message.content
            
            # Store user's image message in conversation history
            self.conversation_memory.add_message(sender, "user", f"[Image sent] {image_description}", image_url)
            
            # Get relevant context from knowledge base
            context = self.knowledge_base.get_relevant_context(image_description)
            
            # Build contextual response using conversation history
            context_messages = self._build_conversation_context(sender, f"Analyze this image and provide a solution: {image_description}")
            
            # Add knowledge base context to the system message
            context_messages[0]["content"] += f"\n\nRelevant technical documentation:\n{context}"
            
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=context_messages,
                max_tokens=500
            )
            
            response_text = f"📷 Analysis complete!\n\nIssue: {image_description}\n\n{completion.choices[0].message.content}"
            
            # Store assistant's response in conversation history
            self.conversation_memory.add_message(sender, "assistant", response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return "Sorry, I couldn't process the image. Please try again or describe the issue in text."

    async def process_text(self, text: str, sender: str) -> str:
        try:
            # Handle special commands
            if text.lower().strip() in ['/clear', '/reset', 'clear history', 'reset conversation']:
                self.conversation_memory.clear_user_history(sender)
                return "✅ Conversation history cleared. How can I help you with your technical issue?"
            
            # First check if it's a technical query
            technical_check = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Determine if this is a technical support query related to equipment or systems. Respond only with 'true' or 'false'."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            is_technical = technical_check.choices[0].message.content.lower().strip() == "true"

            if not is_technical:
                response = "I am a technical support assistant. I can only help with technical issues related to equipment and systems. For other types of questions, please consult the appropriate specialist."
                # Don't store non-technical queries in conversation history
                return response

            # Store user message in conversation history
            self.conversation_memory.add_message(sender, "user", text)

            # Get relevant context from knowledge base
            context = self.knowledge_base.get_relevant_context(text)
            
            # Build conversation context with history
            messages = self._build_conversation_context(sender, text)
            
            # Add knowledge base context to the system message
            messages[0]["content"] += f"\n\nRelevant technical documentation:\n{context}"
            
            # Get AI response with conversation context
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500
            )
            
            response = completion.choices[0].message.content
            
            # Store assistant response in conversation history
            self.conversation_memory.add_message(sender, "assistant", response)
            
            # Check for hardware issues and send notifications
            user_email = self.get_user_email(sender)
            is_hardware_issue = await self.check_hardware_issue(text)
            
            if is_hardware_issue and settings.NOTIFICATION_EMAIL and self.email_service:
                try:
                    # Include conversation context in maintenance notification
                    history = self.conversation_memory.get_conversation_history(sender)
                    conversation_context = "\n".join([
                        f"{msg['role'].title()}: {msg['content']}" 
                        for msg in history[-5:]  # Last 5 messages for context
                    ])
                    
                    await self.email_service.send_maintenance_notification(
                        settings.NOTIFICATION_EMAIL,
                        text,
                        priority="High" if "urgent" in text.lower() else "Medium",
                        hardware_details={
                            "issue_description": text,
                            "user_contact": sender,
                            "user_email": user_email,
                            "conversation_context": conversation_context
                        }
                    )
                    response += "\n\nI've notified our maintenance team about this hardware issue. They will review the conversation context and address it as soon as possible."
                except Exception as e:
                    logger.error(f"Failed to send maintenance notification: {e}")
            
            # Send satisfaction survey if we have user's email and email service is available
            if user_email and self.email_service:
                try:
                    interaction_id = str(uuid.uuid4())
                    await self.email_service.send_satisfaction_survey(user_email, interaction_id)
                    logger.info(f"Sent satisfaction survey to {user_email} for interaction {interaction_id}")
                except Exception as e:
                    logger.error(f"Failed to send satisfaction survey: {e}")
            else:
                logger.info(f"No email found for WhatsApp number {sender}, skipping satisfaction survey")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return "Sorry, I couldn't process your message. Please try again or contact support directly."

    def start_ngrok(self):
        """Start ngrok tunnel with better error handling"""
        try:
            # Kill any existing ngrok processes
            try:
                ngrok.kill()
                time.sleep(2)
            except:
                pass
            
            # Set auth token
            if settings.NGROK_AUTH_TOKEN:
                pyngrok.conf.get_default().auth_token = settings.NGROK_AUTH_TOKEN
            
            # Connect ngrok tunnel
            self.ngrok_tunnel = ngrok.connect(8000)
            tunnel_url = self.ngrok_tunnel.public_url
            
            logger.info(f"ngrok tunnel established at: {tunnel_url}")
            return tunnel_url
                
        except Exception as e:
            logger.error(f"Error starting ngrok: {str(e)}")
            logger.info("Continuing without ngrok tunnel. You'll need to use a different method for webhook access.")
            return None

    def cleanup(self):
        """Cleanup ngrok resources and perform maintenance"""
        try:
            # Cleanup expired conversations
            self.conversation_memory.cleanup_expired_conversations()
            
            if self.ngrok_tunnel:
                ngrok.disconnect(self.ngrok_tunnel.public_url)
            ngrok.kill()
            logger.info("Cleaned up ngrok tunnel and performed maintenance")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# Initialize the diagnostic assistant
assistant = DiagnosticAssistant()

@app.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        form_data = await request.form()
        logger.info(f"Received webhook data: {dict(form_data)}")
        
        sender = form_data.get("From", "")
        if sender.startswith("whatsapp:"):
            sender = sender[9:]
            
        media_url = form_data.get("MediaUrl0")
        message_body = form_data.get("Body", "")
        
        logger.info(f"Processing message from {sender}")
        
        if media_url:
            response = await assistant.process_image(media_url, sender)
        elif message_body:
            response = await assistant.process_text(message_body, sender)
        else:
            response = "I couldn't process your message. Please send either text or an image."
        
        await assistant.send_whatsapp_message(sender, response)
        
        return {"status": "success", "message": "Message processed"}
        
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Diagnostic Assistant is running",
        "version": "3.1.0",  # Updated version with memory
        "status": "online",
        "features": ["multi-turn memory", "conversation history", "context awareness"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "twilio": bool(settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN),
            "openai": bool(settings.OPENAI_API_KEY),
            "email": bool(settings.SENDGRID_API_KEY),
            "ngrok": bool(assistant.ngrok_tunnel),
            "memory": True
        }
    }

@app.get("/conversation/{user_id}")
async def get_conversation_history(user_id: str):
    """Get conversation history for a specific user (for debugging/admin purposes)"""
    try:
        # Clean the user_id (remove whatsapp: prefix if present)
        clean_user_id = user_id.replace('whatsapp:', '').replace('+', '')
        history = assistant.conversation_memory.get_conversation_history(clean_user_id)
        return {
            "user_id": clean_user_id,
            "conversation_count": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversation/{user_id}")
async def clear_conversation_history(user_id: str):
    """Clear conversation history for a specific user (admin endpoint)"""
    try:
        # Clean the user_id (remove whatsapp: prefix if present)
        clean_user_id = user_id.replace('whatsapp:', '').replace('+', '')
        assistant.conversation_memory.clear_user_history(clean_user_id)
        return {
            "message": f"Conversation history cleared for user: {clean_user_id}",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/maintenance/cleanup")
async def manual_cleanup():
    """Manual cleanup of expired conversations (admin endpoint)"""
    try:
        assistant.conversation_memory.cleanup_expired_conversations()
        return {
            "message": "Cleanup completed successfully",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error during manual cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    try:
        print("\n=== Domain-Specific Diagnostic Assistant Starting ===")
        print("Version: 3.1.0 (Enhanced with Multi-turn Memory)")
        print("\nNew Features:")
        print("- ✅ Multi-turn conversation memory")
        print("- ✅ Context-aware responses")
        print("- ✅ Conversation history storage")
        print("- ✅ Automatic conversation cleanup")
        print("- ✅ Memory management commands (/clear, /reset)")
        print(f"- ✅ Remembers last {settings.MAX_CONVERSATION_HISTORY} messages per user")
        print(f"- ✅ Conversations expire after {settings.CONVERSATION_TIMEOUT_HOURS} hours")
        
        # Start ngrok
        public_url = assistant.start_ngrok()
        if public_url:
            print(f"\nWebhook URL: {public_url}/webhook")
            print("\nMake sure to:")
            print("1. Set this webhook URL in your Twilio WhatsApp Sandbox")
            print("2. Test by sending both text and images to your Twilio WhatsApp number")
            print("3. Try follow-up questions to test memory functionality")
        else:
            print("\nNgrok tunnel failed to start. Running in local mode.")
            print("You'll need to use a different tunneling service or deploy to a server.")
        
        print("\nAPI Endpoints:")
        print("- Health check: http://localhost:8000/health")
        print("- Get conversation: http://localhost:8000/conversation/{user_id}")
        print("- Clear conversation: DELETE http://localhost:8000/conversation/{user_id}")
        print("- Manual cleanup: POST http://localhost:8000/maintenance/cleanup")
        
        print("\nUser Commands:")
        print("- '/clear' or '/reset' - Clear conversation history")
        print("- 'clear history' - Clear conversation history")
        
        print("\nMemory Features:")
        print("- The assistant now remembers previous conversations")
        print("- Follow-up questions will reference earlier messages")
        print("- Image analysis includes conversation context")
        print("- Hardware notifications include conversation history")
        
        print("\nPress Ctrl+C to shutdown gracefully")
        
        # Start the FastAPI application
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nError starting application: {str(e)}")
        logger.error(f"Application startup error: {str(e)}")
    finally:
        assistant.cleanup()
        print("Application shutdown complete")

if __name__ == "__main__":
    main()