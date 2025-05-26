
## Smart IT Help Desk

An AI‑powered support assistant written in **Python** (FastAPI) that uses **Twilio WhatsApp** for messaging, **OpenAI GPT‑4** for diagnostics, **Qdrant** for semantic context retrieval, and **SendGrid** for email alerts and surveys. It automates ticket triage, hardware‐issue escalation, and continuous feedback‐driven learning.

---

## 🚀 Features

* **WhatsApp Integration**: Real‑time messaging via Twilio’s WhatsApp Sandbox 
* **AI‑Powered Analysis**: GPT‑4 handles technical issue classification and resolution 
* **Semantic Search**: Qdrant vector database retrieves relevant docs and past tickets 
* **Automated Escalation**: Detects hardware issues and emails maintenance via SendGrid 
* **Feedback Loop**: Sends satisfaction surveys, logs responses for continuous improvement to make the system smarter over time 

---

## 📦 Technology Stack

| Component       | Technology                   | Purpose                                   |
| --------------- | ---------------------------- | ----------------------------------------- |
| Web Framework   | Python 3.9+ & FastAPI        | HTTP server & webhook handling            |
| Messaging       | Twilio WhatsApp API          | Inbound/outbound chat                     |
| AI              | OpenAI GPT‑4 (via LangChain) | Issue analysis & response generation      |
| Vector Database | Qdrant (local/Cloud)         | Semantic context retrieval                |
| Email Service   | Twilio SendGrid API          | Maintenance alerts & satisfaction surveys |
| Tunnel          | ngrok (via pyngrok)          | Expose localhost to Twilio webhooks       |

---

## 📋 Prerequisites

* **Python 3.9+** installed 
* **Twilio account** with WhatsApp Sandbox enabled 
* **OpenAI API key** (GPT‑4 access)
* **SendGrid account** & verified sender email
* **Qdrant** instance
* **ngrok** account for local webhook tunneling 

---

## 🔧 Installation & Setup

1. **Clone the repository**

   ````bash
   git clone https://github.com/lizpart/smart-it-helpdesk.git
   cd smart-it-helpdesk
 
   ````

2. **Create & activate virtual environment**

   * **macOS / Linux**

     ```bash
     python3 -m venv venv  
     source venv/bin/activate  
     ```
   * **Windows (PowerShell)**

     ```powershell
     python -m venv venv  
     .\venv\Scripts\Activate.ps1  
     ```

3. **Install dependencies**

   ````bash
   pip install -r requirements.txt
   

   ````

4. **Configure environment variables**
   Create a `.env` file in project root:

   ```dotenv
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  
   TWILIO_AUTH_TOKEN=your_auth_token  
   TWILIO_PHONE_NUMBER=+11234567890  
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx  
   NGROK_AUTH_TOKEN=your_ngrok_token  
   SENDGRID_API_KEY=SG.xxxxxxxxxxxxxxxxxxxxxx  
   FROM_EMAIL=you@domain.com  
   NOTIFICATION_EMAIL=it-team@company.com  
   USER_EMAILS={"1234567890":"alice@example.com"}  
   ```

   > *Use environment variables to keep secrets out of source control* 


## ▶️ Running Locally

1. **Launch the application**

   ```bash
   python app.py
   ```

   This spins up FastAPI on port 8000 and an ngrok tunnel; note the printed webhook URL.

2. **Configure Twilio Sandbox**
   In Twilio Console → Messaging → WhatsApp Sandbox, set **When a message comes in** to:

   ````
   https://<your-ngrok-id>.ngrok.io/webhook
    

   ````

3. **Test the bot**

   * Send a WhatsApp message (text or image) to the sandbox number.
   * Observe AI‑driven reply, automated escalation emails, and survey follow‑ups.

---

## 📂 Project Structure

```
smart-it-helpdesk/
├── app.py                 # FastAPI app & Twilio webhook handlers  
├── email_service.py       # SendGrid email helper class  
├── technical_docs/        # .txt files ingested into Qdrant  
├── requirements.txt       # Python dependencies  
├── .env                   # Environment variables (not in VCS)  
└── README.md              # Project documentation  
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/YourFeature`
3. Commit: `git commit -m "Add YourFeature"`
4. Push: `git push origin feature/YourFeature`
5. Open a Pull Request

Please follow code style, write tests for new features, and update documentation accordingly.

---

## 📄 License

This project is licensed under the **MIT License**.

---

By following this README, developers—from beginners to experts—can quickly understand, clone, configure, and run your Smart IT Help Desk end‑to‑end.
