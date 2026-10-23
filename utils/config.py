"""Environment-based configuration for both local runs and Docker."""
import os
from dotenv import load_dotenv

load_dotenv()  # Optional locally; Docker Compose injects the environment variables.

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "kkbox_app"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "kkbox_dwh"),
}
