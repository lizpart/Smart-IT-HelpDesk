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
                    <p style="margin-bottom: 20px;">{html.escape(issue_description)}</p>
                    
                    <h3 style="color: #495057;">Priority Level</h3>
                    <p style="font-weight: bold; color: {'#dc3545' if priority == 'High' else '#ffc107'};">
                        {priority}
                    </p>
                    
                    <h3 style="color: #495057;">Contact Information</h3>
                    <ul style="list-style: none; padding: 0;">
                        <li>📱 <strong>WhatsApp Number:</strong> {html.escape(user_contact)}</li>
                        <li>📧 <strong>User Email:</strong> {html.escape(user_email)}</li>
                    </ul>
                    
                    <h3 style="color: #495057;">Hardware Details</h3>
                    <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;">
{html.escape(json.dumps(hardware_details, indent=2) if hardware_details else 'Not specified')}
                    </pre>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 15px; font-size: 0.9em; color: #6c757d;">
                    <p style="margin: 0;">
                        🕒 Time Reported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
            </body>
            </html>
            """
            
            message = Mail(
                from_email=self.from_email,
                to_emails=to_email,
                subject=subject,
                html_content=content
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
            
            # Create an HTML template with mailto links for rating
            survey_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; margin-bottom: 20px;">
                    <h2 style="color: #007bff; margin: 0;">Your Feedback Matters!</h2>
                </div>
                
                <div style="background-color: #fff; border: 1px solid #dee2e6; border-radius: 4px; padding: 20px; margin-bottom: 20px;">
                    <p>We recently helped you with a technical issue and would love to know about your experience.</p>
                    
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 4px; margin-bottom: 20px;">
                        <p style="margin: 0;"><strong>Reference Information:</strong></p>
                        <ul style="list-style: none; padding: 0;">
                            <li>🔍 Interaction ID: {interaction_id}</li>
                            {f'<li>📱 WhatsApp Number: {whatsapp_number}</li>' if whatsapp_number else ''}
                        </ul>
                    </div>
                    
                    <p>Please rate our service by clicking one of the following options:</p>
                    
                    <div style="display: flex; flex-direction: column; gap: 10px;">
                        <a href="mailto:{self.notification_email}?subject=Service%20Rating%20-%20{interaction_id}&body=Rating:%205%20stars%0AInteraction%20ID:%20{interaction_id}" 
                           style="background-color: #28a745; color: white; padding: 10px; text-decoration: none; border-radius: 4px; text-align: center;">
                           ⭐⭐⭐⭐⭐ Excellent
                        </a>
                        
                        <a href="mailto:{self.notification_email}?subject=Service%20Rating%20-%20{interaction_id}&body=Rating:%204%20stars%0AInteraction%20ID:%20{interaction_id}" 
                           style="background-color: #17a2b8; color: white; padding: 10px; text-decoration: none; border-radius: 4px; text-align: center;">
                           ⭐⭐⭐⭐ Very Good
                        </a>
                        
                        <a href="mailto:{self.notification_email}?subject=Service%20Rating%20-%20{interaction_id}&body=Rating:%203%20stars%0AInteraction%20ID:%20{interaction_id}" 
                           style="background-color: #ffc107; color: black; padding: 10px; text-decoration: none; border-radius: 4px; text-align: center;">
                           ⭐⭐⭐ Good
                        </a>
                        
                        <a href="mailto:{self.notification_email}?subject=Service%20Rating%20-%20{interaction_id}&body=Rating:%202%20stars%0AInteraction%20ID:%20{interaction_id}" 
                           style="background-color: #fd7e14; color: white; padding: 10px; text-decoration: none; border-radius: 4px; text-align: center;">
                           ⭐⭐ Fair
                        </a>
                        
                        <a href="mailto:{self.notification_email}?subject=Service%20Rating%20-%20{interaction_id}&body=Rating:%201%20star%0AInteraction%20ID:%20{interaction_id}" 
                           style="background-color: #dc3545; color: white; padding: 10px; text-decoration: none; border-radius: 4px; text-align: center;">
                           ⭐ Needs Improvement
                        </a>
                    </div>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 15px; font-size: 0.9em; color: #6c757d; text-align: center;">
                    <p style="margin: 0;">Thank you for helping us improve our service!</p>
                </div>
            </body>
            </html>
            """
            
            message = Mail(
                from_email=self.from_email,
                to_emails=to_email,
                subject=subject,
                html_content=survey_content
            )
            
            # Disable click tracking as we're using mailto links
            tracking_settings = TrackingSettings()
            tracking_settings.click_tracking = ClickTracking(enable=False)
            message.tracking_settings = tracking_settings
            
            response = self.client.send(message)
            logger.info(f"Satisfaction survey sent successfully to {to_email}. Status code: {response.status_code}")
            return response.status_code
            
        except Exception as e:
            logger.error(f"Error sending satisfaction survey: {str(e)}")
            raise