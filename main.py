import os
import logging
from flask import Flask, request
import telebot
from openai import OpenAI

# Environment variables
BOT_TOKEN = os.environ.get("BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = telebot.TeleBot(BOT_TOKEN)

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

app = Flask(__name__)


def ask_deepseek(user_message: str) -> str:
    """Send a message to DeepSeek-R1 via HuggingFace router and return the response."""
    try:
        chat_completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1:novita",
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling HuggingFace API: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    welcome_text = (
        "👋 Hello! I'm powered by *DeepSeek-R1* via HuggingFace.\n\n"
        "Just send me any message and I'll respond using AI! 🤖"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text
    chat_id = message.chat.id

    # Show typing indicator
    bot.send_chat_action(chat_id, "typing")

    logger.info(f"Received message from {chat_id}: {user_text}")

    response = ask_deepseek(user_text)

    bot.reply_to(message, response)


# Webhook route for Render
@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    json_data = request.get_json()
    update = telebot.types.Update.de_json(json_data)
    bot.process_new_updates([update])
    return "OK", 200


@app.route("/", methods=["GET"])
def index():
    return "Bot is running!", 200


if __name__ == "__main__":
    # Set webhook URL — set RENDER_URL in environment variables on Render
    RENDER_URL = os.environ.get("RENDER_URL")  # e.g. https://your-app.onrender.com

    if RENDER_URL:
        webhook_url = f"{RENDER_URL}/{BOT_TOKEN}"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        logger.info(f"Webhook set to: {webhook_url}")
    else:
        logger.warning("RENDER_URL not set. Webhook not configured.")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
