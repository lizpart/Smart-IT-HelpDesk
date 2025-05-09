import os
import time
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import openai
import logging
from datetime import datetime
import uvicorn
import requests
import base64
from pydantic_settings import BaseSettings
from typing import Optional
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import pyngrok.conf
from pyngrok import ngrok
import uuid
from email_service import EmailService

# Load environment variables
load_dotenv()

# Configure logging
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
    
    class Config:
        env_file = ".env"
        extra = "allow" 

settings = Settings()

# Configure ngrok
pyngrok.conf.get_default().auth_token = settings.NGROK_AUTH_TOKEN

# Initialize FastAPI app
app = FastAPI()

class KnowledgeBase:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        self.collection_name = "technical_docs"
        self.vector_size = 1536
        self.qdrant_client = QdrantClient(path="./qdrant_storage")
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
            raise

    def _create_knowledge_base(self):
        try:
            loader = DirectoryLoader("technical_docs/", glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            for i, doc in enumerate(texts):
                embedding = self.embeddings.embed_query(doc.page_content)
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[models.PointStruct(id=i, vector=embedding, payload={"text": doc.page_content})]
                )
        except Exception as e:
            logger.error(f"Error creating knowledge base: {str(e)}")
            raise

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
            return ""

class DiagnosticAssistant:
    def __init__(self):
        self.twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        self.openai_client = openai.Client(api_key=settings.OPENAI_API_KEY)
        self.knowledge_base = KnowledgeBase()
        self.ngrok_tunnel = None
        self.email_service = EmailService(settings.SENDGRID_API_KEY, settings.FROM_EMAIL, settings.NOTIFICATION_EMAIL)
        self.user_emails = settings.USER_EMAILS

    def get_user_email(self, whatsapp_number: str) -> Optional[str]:
        """Get user email from WhatsApp number"""
        # Remove 'whatsapp:' prefix if present
        clean_number = whatsapp_number.replace('whatsapp:', '')
        return self.user_emails.get(clean_number)

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
                model="gpt-4",
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
                max_tokens=10
            )
            response = completion.choices[0].message.content.lower().strip()
            return response == "true"
            
        except Exception as e:
            logger.error(f"Error checking hardware issue: {str(e)}")
            return False

    async def process_image(self, image_url: str) -> str:
        try:
            response = requests.get(
                image_url,
                auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
                timeout=30
            )
            response.raise_for_status()
            
            image_data = base64.b64encode(response.content).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{image_data}"
            
            vision_completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this technical issue or error message:"},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                max_tokens=300
            )
            
            image_description = vision_completion.choices[0].message.content
            context = self.knowledge_base.get_relevant_context(image_description)
            
            completion = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical support assistant. Provide clear, concise solutions."
                    },
                    {
                        "role": "user",
                        "content": f"Using this context:\n{context}\n\nProvide a solution for this issue: {image_description}"
                    }
                ],
                max_tokens=500
            )
            
            return f"📷 Analysis complete!\n\nIssue: {image_description}\n\n{completion.choices[0].message.content}"
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return "Sorry, I couldn't process the image. Please try again."

    async def process_text(self, text: str, sender: str) -> str:
        try:
            # First check if it's a technical query
            technical_check = self.openai_client.chat.completions.create(
                model="gpt-4",
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
                max_tokens=10
            )
            is_technical = technical_check.choices[0].message.content.lower().strip() == "true"

            if not is_technical:
                return "I am a technical support assistant. I can only help with technical issues related to equipment and systems. For other types of questions, please consult the appropriate specialist."

            # Continue with regular processing for technical queries
            context = self.knowledge_base.get_relevant_context(text)
            user_email = self.get_user_email(sender)
            is_hardware_issue = await self.check_hardware_issue(text)
            
            completion = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical support assistant. Provide clear, concise solutions."
                    },
                    {
                        "role": "user",
                        "content": f"Using this context:\n{context}\n\nAnswer this question: {text}"
                    }
                ],
                max_tokens=500
            )
            
            response = completion.choices[0].message.content
            
            # If it's a hardware issue, send maintenance notification
            if is_hardware_issue and settings.NOTIFICATION_EMAIL:
                await self.email_service.send_maintenance_notification(
                    settings.NOTIFICATION_EMAIL,
                    text,
                    priority="High" if "urgent" in text.lower() else "Medium",
                    hardware_details={
                        "issue_description": text,
                        "user_contact": sender,
                        "user_email": user_email
                    }
                )
                response += "\n\nI've notified our maintenance team about this hardware issue. They will address it as soon as possible."
            
            # Send satisfaction survey if we have user's email
            if user_email:
                interaction_id = str(uuid.uuid4())
                await self.email_service.send_satisfaction_survey(user_email, interaction_id)
                logger.info(f"Sent satisfaction survey to {user_email} for interaction {interaction_id}")
            else:
                logger.info(f"No email found for WhatsApp number {sender}, skipping satisfaction survey")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return "Sorry, I couldn't process your message. Please try again."

    def start_ngrok(self):
        """Start ngrok tunnel"""
        try:
            # Set auth token
            pyngrok.conf.get_default().auth_token = settings.NGROK_AUTH_TOKEN
            
            # Connect directly without killing previous processes
            self.ngrok_tunnel = ngrok.connect(8000)
            tunnel_url = self.ngrok_tunnel.public_url
            
            logger.info(f"ngrok tunnel established at: {tunnel_url}")
            return tunnel_url
                
        except Exception as e:
            logger.error(f"Error starting ngrok: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup ngrok resources"""
        try:
            if self.ngrok_tunnel:
                ngrok.disconnect(self.ngrok_tunnel.public_url)
            logger.info("Cleaned up ngrok tunnel")
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
            response = await assistant.process_image(media_url)
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
        "version": "2.0.0",
        "status": "online"
    }

def main():
    try:
        print("\n=== Domain-Specific Diagnostic Assistant Starting ===")
        
        # Start ngrok
        public_url = assistant.start_ngrok()
        print(f"\nWebhook URL: {public_url}/webhook")
        print("\nMake sure to:")
        print("1. Set this webhook URL in your Twilio WhatsApp Sandbox")
        print("2. Test by sending both text and images to your Twilio WhatsApp number")
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