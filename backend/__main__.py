"""One-command launch: ``python -m backend`` (serves API + built dashboard).

Hosting platforms (Render, Railway, ...) pass the port in ``$PORT``; when it is
set the server also binds to all interfaces.
"""

import argparse
import os
import webbrowser

import uvicorn


def main() -> None:
    hosted = "PORT" in os.environ
    ap = argparse.ArgumentParser(description="Adaptive 2.5D LiDAR mapping prototype")
    ap.add_argument("--host", default="0.0.0.0" if hosted else "127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8053)))
    ap.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = ap.parse_args()
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    uvicorn.run("backend.app:app", host=args.host, port=args.port, log_level="info", proxy_headers=True,
                forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
