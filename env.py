from dotenv import load_dotenv, find_dotenv
import os

def load_dotenv_if_exists():
    try:
        return load_dotenv(find_dotenv())
    except Exception:
        print("No .env file found.")

def load_token(token_name):
    # load_dotenv_if_exists()
    return 1
