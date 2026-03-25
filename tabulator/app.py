import os
from flask import Flask
from asgiref.wsgi import WsgiToAsgi
from .config import BaseConfig, LocalDevConfig, ProductionConfig


def create_flask_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
        instance_relative_config=True,
    )

    # Determine whether to run in local dev mode
    local_dev_flag = os.getenv("LOCAL_DEV", "0").lower() in {"1", "true", "yes"}
    app.config.from_object(LocalDevConfig if local_dev_flag else BaseConfig)

    # Ensure upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.instance_path, exist_ok=True)

    # Register blueprints
    from .routes import bp as main_bp

    app.register_blueprint(main_bp)
    return app


def create_asgi_app():
    return WsgiToAsgi(create_flask_app())


def create_app() -> Flask:
    return create_flask_app()
