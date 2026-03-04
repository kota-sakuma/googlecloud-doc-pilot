"""
Main entry point for the application.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from .mcp_client import fetch_googlecloud_doc
from .engine import generate_answer, translate_to_english

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = App(token=os.getenv("SLACK_BOT_TOKEN"))


@app.message()
def handle_message(message, say):
    """
    Handle a message event from Slack.
    """
    text = message.get("text", "").strip()
    if not text:
        return

    # Question from user (answer will be in the same language)
    user_question = text

    try:
        # Translate to English for document search (docs are indexed in English)
        search_query = translate_to_english(user_question)
        # Fetch documentation from Google Cloud
        context_docs = asyncio.run(fetch_googlecloud_doc(search_query))
        logger.info("Successfully fetched documentation for user question")
        logger.debug("Context docs: %s", context_docs)

        # Generate answer using Gemini API
        answer = generate_answer(user_question, context_docs)
        logger.info("Successfully generated answer for user question")
        logger.debug("Answer: %s", answer)
    except (ValueError, OSError) as e:
        logger.error("Error generating answer: %s", e)
        answer = "エラーが発生しました。もう一度お試しください。"
    finally:
        say(answer)


def run():
    """
    Run the application with Socket Mode.
    """
    app_token = os.getenv("SLACK_APP_TOKEN")
    if not app_token:
        raise ValueError("SLACK_APP_TOKEN is not set")
    handler = SocketModeHandler(app, app_token)
    handler.start()
    logger.info("Application started")


if __name__ == "__main__":
    run()
