from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To, Email, Content, ClickTracking, TrackingSettings
from datetime import datetime
import logging
from typing import List, Dict, Optional
import json
import html

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self, api_key: str, from_email: str, notification_email: str):
        if not api_key:
            raise ValueError("SendGrid API key is required")
        if not from_email:
            raise ValueError("From email is required")
        if not notification_email:
            raise ValueError("Notification email is required")
            
        self.client = SendGridAPIClient(api_key)
        self.from_email = from_email
        self.notification_email = notification_email
        
    async def send_maintenance_notification(
        self, 
        to_email: str, 
        issue_description: str,
        priority: str = "Medium",
        hardware_details: Dict = None
    ):
        try:
            subject = f"🔧 Maintenance Required: {priority} Priority Issue Detected"
            
            # Format user contact information
            user_contact = hardware_details.get('user_contact', 'Not provided') if hardware_details else 'Not provided'
            user_email = hardware_details.get('user_email', 'Not provided') if hardware_details else 'Not provided'
            
            # Create a more structured HTML content
            content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #f8f9fa; border-left: 4px solid #dc3545; padding: 15px; margin-bottom: 20px;">
                    <h2 style="color: #dc3545; margin: 0;">⚠️ Urgent Maintenance Required</h2>
                </div>
                
                <div style="background-color: #fff; border: 1px solid #dee2e6; border-radius: 4px; padding: 20px; margin-bottom: 20px;">
                    <h3 style="color: #495057; margin-top: 0;">Issue Description</h3>
                    <p style="margin-bottom: 20px; background-color: #f8f9fa; padding: 10px; border-radius: 4px;">{html.escape(issue_description)}</p>
                    
                    <h3 style="color: #495057;">Priority Level</h3>
                    <p style="font-weight: bold; color: {'#dc3545' if priority == 'High' else '#ffc107' if priority == 'Medium' else '#28a745'}; font-size: 1.2em;">
                        🚨 {priority.upper()} PRIORITY
                    </p>
                    
                    <h3 style="color: #495057;">Contact Information</h3>
                    <div style="background-color: #e9ecef; padding: 15px; border-radius: 4px;">
                        <p style="margin: 5px 0;"><strong>📱 WhatsApp Number:</strong> {html.escape(user_contact)}</p>
                        <p style="margin: 5px 0;"><strong>📧 User Email:</strong> {html.escape(user_email)}</p>
                    </div>
                    
                    <h3 style="color: #495057;">Additional Details</h3>
                    <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; font-size: 0.9em; border: 1px solid #dee2e6;">
{html.escape(json.dumps(hardware_details, indent=2) if hardware_details else 'No additional details provided')}
                    </pre>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; font-size: 0.9em; color: #6c757d; text-align: center;">
                    <p style="margin: 0;">
                        🕒 <strong>Time Reported:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
                    </p>
                    <p style="margin: 5px 0 0 0; font-style: italic;">
                        This notification was generated automatically by the Diagnostic Assistant system.
                    </p>
                </div>
            </body>
            </html>
            """
            
            message = Mail(
                from_email=Email(self.from_email),
                to_emails=[To(to_email)],
                subject=subject,
                html_content=Content("text/html", content)
            )
            
            response = self.client.send(message)
            logger.info(f"Maintenance notification sent successfully. Status code: {response.status_code}")
            return response.status_code
            
        except Exception as e:
            logger.error(f"Error sending maintenance notification: {str(e)}")
            raise
    
    async def send_satisfaction_survey(
        self, 
        to_email: str, 
        interaction_id: str, 
        whatsapp_number: Optional[str] = None
    ):
        try:
            subject = "📝 How was your experience? We'd love your feedback!"
            
            # Create survey URL parameters for better tracking
            survey_base_params = f"subject=Service%20Rating%20-%20{interaction_id}&body="
            
            # Create an HTML template with mailto links for rating
            survey_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; margin-bottom: 20px;">
                    <h2 style="color: #007bff; margin: 0;">🌟 Your Feedback Matters!</h2>
                </div>
                
                <div style="background-color: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <p style="font-size: 1.1em; margin-bottom: 20px;">We recently helped you with a technical issue and would love to know about your experience.</p>
                    
                    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 6px; margin-bottom: 25px; border-left: 4px solid #2196f3;">
                        <p style="margin: 0; font-weight: bold; color: #1976d2;">Reference Information:</p>
                        <ul style="list-style: none; padding: 10px 0 0 0; margin: 0;">
                            <li style="margin: 5px 0;">🔍 <strong>Interaction ID:</strong> {interaction_id}</li>
                            {f'<li style="margin: 5px 0;">📱 <strong>WhatsApp Number:</strong> {whatsapp_number}</li>' if whatsapp_number else ''}
                        </ul>
                    </div>
                    
                    <p style="font-size: 1.1em; margin-bottom: 20px; text-align: center;"><strong>Please rate our service by clicking one of the options below:</strong></p>
                    
                    <div style="display: flex; flex-direction: column; gap: 12px; max-width: 400px; margin: 0 auto;">
                        <a href="mailto:{self.notification_email}?{survey_base_params}Rating:%205%20stars%0AInteraction%20ID:%20{interaction_id}%0AComments:%20Please%20add%20any%20additional%20feedback%20here" 
                           style="background: linear-gradient(135deg, #28a745, #34ce57); color: white; padding: 15px 20px; text-decoration: none; border-radius: 8px; text-align: center; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                           ⭐⭐⭐⭐⭐ Excellent - Exceeded Expectations
                        </a>
                        
                        <a href="mailto:{self.notification_email}?{survey_base_params}Rating:%204%20stars%0AInteraction%20ID:%20{interaction_id}%0AComments:%20Please%20add%20any%20additional%20feedback%20here" 
                           style="background: linear-gradient(135deg, #17a2b8, #20c997); color: white; padding: 15px 20px; text-decoration: none; border-radius: 8px; text-align: center; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                           ⭐⭐⭐⭐ Very Good - Met Expectations
                        </a>
                        
                        <a href="mailto:{self.notification_email}?{survey_base_params}Rating:%203%20stars%0AInteraction%20ID:%20{interaction_id}%0AComments:%20Please%20add%20any%20additional%20feedback%20here" 
                           style="background: linear-gradient(135deg, #ffc107, #ffdb4d); color: #212529; padding: 15px 20px; text-decoration: none; border-radius: 8px; text-align: center; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                           ⭐⭐⭐ Good - Satisfactory Service
                        </a>
                        
                        <a href="mailto:{self.notification_email}?{survey_base_params}Rating:%202%20stars%0AInteraction%20ID:%20{interaction_id}%0AComments:%20Please%20add%20any%20additional%20feedback%20here" 
                           style="background: linear-gradient(135deg, #fd7e14, #ff922b); color: white; padding: 15px 20px; text-decoration: none; border-radius: 8px; text-align: center; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                           ⭐⭐ Fair - Room for Improvement
                        </a>
                        
                        <a href="mailto:{self.notification_email}?{survey_base_params}Rating:%201%20star%0AInteraction%20ID:%20{interaction_id}%0AComments:%20Please%20add%20any%20additional%20feedback%20here" 
                           style="background: linear-gradient(135deg, #dc3545, #e85d75); color: white; padding: 15px 20px; text-decoration: none; border-radius: 8px; text-align: center; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                           ⭐ Poor - Needs Significant Improvement
                        </a>
                    </div>
                    
                    <div style="margin-top: 25px; padding: 15px; background-color: #f8f9fa; border-radius: 6px; text-align: center;">
                        <p style="margin: 0; font-size: 0.95em; color: #6c757d;">
                            💡 <strong>Tip:</strong> Click any rating above to open your email client with a pre-filled message. 
                            Feel free to add additional comments in the email body!
                        </p>
                    </div>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; font-size: 0.9em; color: #6c757d; text-align: center;">
                    <p style="margin: 0 0 10px 0; font-weight: bold;">Thank you for helping us improve our service! 🙏</p>
                    <p style="margin: 0; font-style: italic;">
                        This survey was sent automatically. Your feedback helps us provide better technical support.
                    </p>
                </div>
            </body>
            </html>
            """
            
            message = Mail(
                from_email=Email(self.from_email),
                to_emails=[To(to_email)],
                subject=subject,
                html_content=Content("text/html", survey_content)
            )
            
            # Disable click tracking as we're using mailto links
            tracking_settings = TrackingSettings()
            tracking_settings.click_tracking = ClickTracking(enable=False, enable_text=False)
            message.tracking_settings = tracking_settings
            
            response = self.client.send(message)
            logger.info(f"Satisfaction survey sent successfully to {to_email}. Status code: {response.status_code}")
            return response.status_code
            
        except Exception as e:
            logger.error(f"Error sending satisfaction survey: {str(e)}")
            raise

    async def send_test_email(self, to_email: str) -> bool:
        """Send a test email to verify email service is working"""
        try:
            subject = "🧪 Test Email - Diagnostic Assistant"
            content = """
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; padding: 15px; margin-bottom: 20px;">
                    <h2 style="color: #155724; margin: 0;">✅ Email Service Test Successful!</h2>
                </div>
                
                <p>This is a test email from the Diagnostic Assistant system to verify that email functionality is working correctly.</p>
                
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0;">
                    <p style="margin: 0;"><strong>Test Details:</strong></p>
                    <ul>
                        <li>Service: SendGrid Email API</li>
                        <li>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</li>
                        <li>Status: Operational</li>
                    </ul>
                </div>
                
                <p style="color: #6c757d; font-size: 0.9em; text-align: center; margin-top: 30px;">
                    If you received this email, the email service is configured correctly.
                </p>
            </body>
            </html>
            """
            
            message = Mail(
                from_email=Email(self.from_email),
                to_emails=[To(to_email)],
                subject=subject,
                html_content=Content("text/html", content)
            )
            
            response = self.client.send(message)
            logger.info(f"Test email sent successfully. Status code: {response.status_code}")
            return response.status_code == 202
            
        except Exception as e:
            logger.error(f"Error sending test email: {str(e)}")
            return False