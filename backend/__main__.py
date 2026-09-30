"""One-command launch: ``python -m backend`` (serves API + built dashboard)."""

import argparse
import webbrowser

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="Adaptive 2.5D LiDAR mapping prototype")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8053)
    ap.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = ap.parse_args()
    if args.open:
        webbrowser.open(f"http://{args.host}:{args.port}")
    uvicorn.run("backend.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
