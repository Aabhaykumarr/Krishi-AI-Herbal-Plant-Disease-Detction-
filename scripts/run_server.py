"""Server entrypoint for KrishiAI FastAPI service (Section 51)."""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="KrishiAI FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    print(f"Starting KrishiAI API Server on http://{args.host}:{args.port}")
    uvicorn.run(
        "krishiai.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
