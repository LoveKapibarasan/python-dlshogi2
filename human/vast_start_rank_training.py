#!/usr/bin/env python3
"""Provision one Vast.ai GPU to train the rank-filtered models from R2.

Required environment: VAST_API_TOKEN, R2_ACCESS_KEY,
R2_SECRET_ACCESS_KEY, and R2_ENDPOINT. The R2 credentials stay local; the
instance receives short-lived, object-scoped signed URLs instead.
"""

import argparse
import datetime
import hashlib
import hmac
import json
import os
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

VAST_API = "https://console.vast.ai/api/v0"
GITHUB_RAW = "https://raw.githubusercontent.com/LoveKapibarasan/python-dlshogi2"


def api_request(url, method="GET", data=None, headers=None):
    body = None if data is None else json.dumps(data).encode()
    request_headers = dict(headers or {})
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    req = Request(url, data=body, headers=request_headers, method=method)
    with urlopen(req, timeout=90) as response:
        payload = response.read()
        return json.loads(payload) if payload else {}


def signed_url(method, object_key, *, endpoint, bucket, access_key, secret_key, expires):
    endpoint_parts = urlsplit(endpoint.rstrip("/"))
    host = endpoint_parts.netloc
    path = f"{endpoint_parts.path.rstrip('/')}/{bucket}/{object_key.lstrip('/')}"
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    date = now.strftime("%Y%m%d")
    scope = f"{date}/auto/s3/aws4_request"
    params = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key}/{scope}",
        "X-Amz-Date": stamp,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_query = urlencode(sorted(params.items()), quote_via=quote, safe="~-")
    canonical_request = (
        f"{method}\n{path}\n{canonical_query}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD"
    )
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{stamp}\n{scope}\n"
        f"{hashlib.sha256(canonical_request.encode()).hexdigest()}"
    )

    def sign(key, message):
        return hmac.new(key, message.encode(), hashlib.sha256).digest()

    signing_key = sign(("AWS4" + secret_key).encode(), date)
    for component in ("auto", "s3", "aws4_request"):
        signing_key = sign(signing_key, component)
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return (
        f"{endpoint_parts.scheme}://{host}{path}?{canonical_query}"
        f"&X-Amz-Signature={signature}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", required=True, help="R2 object key for the HCPE tar bundle")
    parser.add_argument("--result-key", required=True, help="R2 object key to receive the results tar.gz")
    parser.add_argument("--bucket", default="assets")
    parser.add_argument("--branch", default="human-like")
    parser.add_argument("--label", default="human-rank-training")
    parser.add_argument("--gpu", default="Tesla V100")
    parser.add_argument("--offer-id", type=int, help="Use a specific offer instead of searching")
    parser.add_argument("--max-hourly", type=float, default=0.25)
    parser.add_argument("--disk-gb", type=int, default=40)
    parser.add_argument("--expires", type=int, default=604800, help="Signed URL lifetime in seconds (max 604800)")
    parser.add_argument("--bands", default="0029-0029 0031-0031 0033-0033")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    token = os.environ["VAST_API_TOKEN"]
    r2_access = os.environ["R2_ACCESS_KEY"]
    r2_secret = os.environ["R2_SECRET_ACCESS_KEY"]
    r2_endpoint = os.environ["R2_ENDPOINT"]
    if not 1 <= args.expires <= 604800:
        parser.error("--expires must be between 1 and 604800 seconds")

    headers = {"Authorization": f"Bearer {token}"}
    dataset_url = signed_url(
        "GET", args.dataset_key, endpoint=r2_endpoint, bucket=args.bucket,
        access_key=r2_access, secret_key=r2_secret, expires=args.expires,
    )
    result_url = signed_url(
        "PUT", args.result_key, endpoint=r2_endpoint, bucket=args.bucket,
        access_key=r2_access, secret_key=r2_secret, expires=args.expires,
    )

    existing = api_request(f"{VAST_API}/instances/", headers=headers).get("instances", [])
    for instance in existing:
        if instance.get("label") == args.label and instance.get("actual_status") not in {"exited", "deleting"}:
            print(f"existing_instance={instance['id']} status={instance.get('actual_status')}")
            return

    offer = None
    if args.offer_id is not None:
        offer = {"id": args.offer_id}
    else:
        query = {
            "gpu_name": {"in": [args.gpu]},
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "reliability": {"gte": 0.95},
            "type": "on-demand",
            "limit": 100,
        }
        offers = api_request(f"{VAST_API}/bundles/", "POST", query, headers).get("offers", [])
        candidates = [
            item for item in offers
            if item.get("disk_space", 0) >= args.disk_gb
            and item.get("dph_total", float("inf")) <= args.max_hourly
        ]
        if not candidates:
            raise RuntimeError("No verified rentable offer matches the GPU, disk, and hourly limit")
        offer = min(candidates, key=lambda item: item["dph_total"])

    script_url = f"{GITHUB_RAW}/{args.branch}/human/vast_cloud_train.sh"
    onstart = (
        "mkdir -p /workspace; "
        f"curl -fsSL '{script_url}' -o /workspace/vast-cloud-train.sh; "
        "chmod +x /workspace/vast-cloud-train.sh; "
        f"nohup env DATA_BUNDLE_URL='{dataset_url}' RESULT_UPLOAD_URL='{result_url}' "
        f"DATA_DIR=/workspace/human_data_ranks/20260926 BANDS='{args.bands}' "
        f"EPOCHS={args.epochs} /workspace/vast-cloud-train.sh "
        "> /workspace/launcher.log 2>&1 < /dev/null &"
    )
    body = {
        "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        "disk": args.disk_gb,
        "label": args.label,
        "runtype": "ssh_direct",
        "cancel_unavail": True,
        "onstart": onstart,
    }
    created = api_request(
        f"{VAST_API}/asks/{offer['id']}/", "PUT", body, headers
    )
    instance_id = created.get("new_contract")
    if not instance_id:
        raise RuntimeError("Vast.ai did not return an instance ID")
    hourly = offer.get("dph_total", "unknown")
    print(f"created_instance={instance_id} offer={offer['id']} hourly_rate={hourly}")


if __name__ == "__main__":
    main()
