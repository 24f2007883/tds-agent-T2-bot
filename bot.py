import os
import json
import logging
import re
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from openai import OpenAI

# Logging configuration
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# Environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
LOG_URL = os.environ.get("LOG_URL", "https://storage.googleapis.com/q1-0b658cf4c1453ef/run.jsonl")

# NVIDIA API Client Setup
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Multi-turn memory
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
        # Fast & Reliable Llama 3.1 8B Model
        response = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=CHAT_HISTORIES[chat_id],
            temperature=0.1,
            max_tokens=1024
        )

        raw_content = response.choices[0].message.content.strip()
        logging.info(f"Model raw output: {raw_content}")

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

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Bot running with Llama-3.1-8b (Fast Response)...")
    app.run_polling()

if __name__ == "__main__":
    main()