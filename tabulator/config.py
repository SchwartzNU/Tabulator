import os


BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_UPLOADS_PATH = os.path.join(BASE_DIR, "uploads")


class BaseConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 100 * 1024 * 1024))  # 100 MB
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", DEFAULT_UPLOADS_PATH)
    ALLOWED_EXTENSIONS = {"csv", "h5", "pkl", "pickle", "mat"}


class LocalDevConfig(BaseConfig):
    DEBUG = True
    ENV = "development"
    TEMPLATES_AUTO_RELOAD = True
    LOCAL_DEV = True

