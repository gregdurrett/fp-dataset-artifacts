import os
import openai

from dotenv import load_dotenv


def init_openai():
    load_dotenv()  # take environment variables from .env.
    openai.api_key = os.getenv('OPENAI_API_KEY')
