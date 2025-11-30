import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut
from pydantic import BaseModel

from hybrid_qa import HybridQAPipeline

# ------------------------
# Load env variables
# ------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # set in .env

if not BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment")

# ------------------------
# Telegram Bot + QA Pipeline
# ------------------------
app_bot = Application.builder().token(BOT_TOKEN).build()
qa_pipeline = HybridQAPipeline()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process incoming Telegram messages."""
    if not update.message:
        return

    user_message = update.message.text
    result = qa_pipeline.ask(user_message)

    reply = f"🔎 *Route*: `{result.route}`\n\n{result.answer}"

    await update.message.reply_text(reply, parse_mode="Markdown")


app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# ------------------------
# FASTAPI APP (ONE APP)
# ------------------------
app = FastAPI()


@app.get("/")
def home():
    return {"message": "Telegram Webhook + RAG API Running"}


# 👉 TELEGRAM WEBHOOK ENDPOINT
@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, app_bot.bot)
    await app_bot.process_update(update)
    return JSONResponse({"status": "ok"})


# 👉 NORMAL API ENDPOINT FOR RAG/SQL
class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(question: Question):
    result = qa_pipeline.ask(question.question)
    return {"route": result.route, "answer": result.answer}


# ------------------------
# Start-up: Set Telegram Webhook
# ------------------------
@app.on_event("startup")
async def startup_event():
    try:
        await app_bot.initialize()
        await app_bot.bot.delete_webhook(drop_pending_updates=True)
        await app_bot.bot.set_webhook(WEBHOOK_URL)
        print("🚀 Webhook set to:", WEBHOOK_URL)
    except TimedOut:
        print("⚠️ Telegram webhook setup timed out (retry later).")
    except Exception as e:
        print(f"⚠️ Webhook setup error: {e}")