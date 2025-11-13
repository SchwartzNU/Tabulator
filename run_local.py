import os
import uvicorn

def main():
    # Ensure local dev mode by default when running this script
    os.environ.setdefault("LOCAL_DEV", "1")

    uvicorn.run(
        "tabulator:create_app",  # module:function
        host="127.0.0.1",
        port=5001,
        reload=True,
        factory=True,            # tells uvicorn this is a factory
    )


if __name__ == "__main__":
    main()