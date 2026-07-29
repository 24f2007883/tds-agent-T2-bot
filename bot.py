import os
import json
import logging
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from openai import OpenAI

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
LOG_URL = os.environ.get("LOG_URL", "https://storage.googleapis.com/q2-9b42d5a2f0465be/run.jsonl")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

CHAT_HISTORIES = {}

SYSTEM_PROMPT = f"""You are a Data Analyst AI agent.

RULES:
1. Answer data analysis questions accurately.
2. If requested to reply with 'OK', respond ONLY with 'OK'.
3. Your FINAL response MUST ALWAYS BE A VALID JSON OBJECT ONLY with two keys:
   - "answer": <the requested answer shape/object>
   - "log_url": "{LOG_URL}"

Do NOT include markdown formatting like ```json ... ```, output raw JSON only.
"""

def clean_and_parse_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text).strip()
    if "FINAL_ANSWER:" in text:
        text = text.split("FINAL_ANSWER:", 1)[1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    logging.info(f"Chat {chat_id} received: {user_text}")

    if chat_id not in CHAT_HISTORIES:
        CHAT_HISTORIES[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    CHAT_HISTORIES[chat_id].append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=CHAT_HISTORIES[chat_id],
            temperature=0.1,
            max_tokens=1024
        )

        raw_content = response.choices[0].message.content.strip()
        CHAT_HISTORIES[chat_id].append({"role": "assistant", "content": raw_content})

        if raw_content.strip().upper() == "OK":
            await update.message.reply_text("OK")
            return

        try:
            parsed_answer = clean_and_parse_json(raw_content)
        except Exception:
            parsed_answer = {"result": raw_content}

        if isinstance(parsed_answer, dict) and "answer" in parsed_answer:
            final_response = parsed_answer
            final_response["log_url"] = LOG_URL
        else:
            final_response = {
                "answer": parsed_answer,
                "log_url": LOG_URL
            }

        await update.message.reply_text(json.dumps(final_response))

    except Exception as e:
        logging.error(f"Error handling request: {e}")
        fallback_response = {
            "answer": "Error processing request",
            "log_url": LOG_URL
        }
        await update.message.reply_text(json.dumps(fallback_response))

class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot Active")

def run_dummy_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), DummyHandler)
    server.serve_forever()

def main():
    threading.Thread(target=run_dummy_server, daemon=True).start()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
