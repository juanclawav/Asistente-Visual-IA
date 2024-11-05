import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Directories for input images and mixed data
    INPUT_IMAGE_PATH = Path("./ai_assistant/data/input_images")
    DATA_PATH = Path("./ai_assistant/data/mixed_wiki")

    @staticmethod
    def prepare_directories():
        for path in [Config.INPUT_IMAGE_PATH, Config.DATA_PATH]:
            path.mkdir(parents=True, exist_ok=True)

# Prepare directories on load
Config.prepare_directories()