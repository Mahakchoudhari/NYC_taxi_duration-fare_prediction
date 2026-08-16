from pathlib import Path
from pydantic_settings import BaseSettings


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    APP_NAME: str = "NYC Taxi Prediction API"
    APP_VERSION: str = "2.0.0"

    MODEL_PATH: str = "Backend/best_model.pkl"
    PIPELINE_PATH: str = "Backend/pipe.pkl"

    FRONTEND_URL: str = "http://localhost:5173"

    ALLOWED_ORIGINS: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173,"
        "http://localhost:5174,"
        "http://127.0.0.1:5174"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()