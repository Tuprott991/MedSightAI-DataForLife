import logging
from app.services.llm_service import medgemma_service
import asyncio
from app.config.database import SessionLocal

import requests
from io import BytesIO
from PIL import Image

def test():
    # we just want to see the error that generate_chat_response throws
    try:
        medgemma_service.generate_chat_response(
            conversation_history=[],
            student_query="What do you see in the image?",
            image_url="https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
