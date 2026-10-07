#!/usr/bin/env python3
"""Fetch container logs for a Vast.ai instance using its API key."""

import argparse
import json
import os
import re
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request_logs(api_key, instance_id, tail):
    url = f"https://console.vast.ai/api/v0/instances/request_logs/{instance_id}/"
    body = json.dumps({"tail": str(tail)}).encode()
    req = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    with urlopen(req, timeout=45) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id", type=int)
    parser.add_argument("--tail", type=int, default=200)
    args = parser.parse_args()
    api_key = os.environ["VAST_API_TOKEN"]

    try:
        result = request_logs(api_key, args.instance_id, args.tail)
    except HTTPError as exc:
        raise SystemExit(f"Vast log request failed: HTTP {exc.code}") from None
    if not result.get("success"):
        raise SystemExit(f"Vast log request failed: {result.get('msg', 'unknown error')}")

    download_url = result.get("temp_download_url") or result.get("result_url")
    if not download_url:
        raise SystemExit("Vast did not return a temporary log download URL")

    for attempt in range(20):
        try:
            with urlopen(download_url, timeout=45) as response:
                lines = response.read().decode("utf-8", errors="replace").splitlines()
            break
        except Exception:
            if attempt == 19:
                raise SystemExit("Vast log download did not become available") from None
            time.sleep(2)

    # Never print temporary URLs or long token-like values from service logs.
    for line in lines[-args.tail :]:
        line = re.sub(r"https?://\S+", "[URL]", line)
        line = re.sub(r"[A-Za-z0-9_=-]{48,}", "[redacted]", line)
        print(line)


if __name__ == "__main__":
    main()
