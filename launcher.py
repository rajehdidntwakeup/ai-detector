"""Open an existing local instance, start a new one, or stop it explicitly."""
import argparse
import http.client
import json
from pathlib import Path
import subprocess
import sys
import webbrowser


def existing_app(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        return response.status == 200 and json.loads(response.read()).get("app") == "local-lens"
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    running = existing_app(args.port)
    if args.stop:
        if not running:
            print("Local Lens is already stopped on this port.")
        else:
            connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=5)
            connection.request("POST", "/api/stop", headers={"X-Local-Lens-Action": "stop"})
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status != 200:
                raise SystemExit("Could not stop the app. Close its server window instead.")
            print("Local Lens stopped.")
    elif running:
        webbrowser.open(f"http://localhost:{args.port}")
        print("Opened the running Local Lens app.")
    else:
        raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).with_name("server.py")), "--port", str(args.port), "--open"]))
