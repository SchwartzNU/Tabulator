import os

from tabulator import create_app


def main() -> None:
    # Ensure local dev mode by default when running this script
    os.environ.setdefault("LOCAL_DEV", "1")
    app = create_app()
    # Explicit host/port for clarity; tweak as desired
    app.run(host="127.0.0.1", port=5001, debug=True)


if __name__ == "__main__":
    main()

