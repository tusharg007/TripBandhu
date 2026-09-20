"""Verify that a deployed TripBandhu build serves the expected release contract."""

from __future__ import annotations

import argparse
import json
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


SMOKE_PLAN_ID = "__tripbandhu_pdf_smoke__"


class SmokeCheckError(RuntimeError):
    """Raised when the public deployment does not match the release contract."""


def _request(url: str, *, payload: dict | None = None, timeout: float = 45.0):
    body = None
    headers = {"User-Agent": "TripBandhu-Deployment-Smoke/1.0"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method="POST" if body else "GET")
    try:
        return urlopen(request, timeout=timeout)
    except HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise SmokeCheckError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise SmokeCheckError(f"Could not reach {url}: {exc.reason}") from exc


def _revisions_match(expected: str, actual: str) -> bool:
    expected = expected.strip().casefold()
    actual = actual.strip().casefold()
    return bool(expected and actual) and (expected.startswith(actual) or actual.startswith(expected))


def verify_deployment(base_url: str, expected_sha: str = "") -> dict[str, object]:
    base_url = base_url.rstrip("/") + "/"

    with _request(urljoin(base_url, "version")) as response:
        version = json.loads(response.read().decode("utf-8"))
        version_header = response.headers.get("X-TripBandhu-Build", "")

    required_fields = {
        "app_version",
        "build_sha",
        "asset_version",
        "pdf_renderer",
        "provider_config_version",
        "public_schema_version",
    }
    missing = sorted(required_fields - set(version))
    if missing:
        raise SmokeCheckError(f"/version is missing fields: {', '.join(missing)}")
    if version["pdf_renderer"] != "reportlab-v1":
        raise SmokeCheckError(f"Unexpected PDF renderer: {version['pdf_renderer']!r}")
    if version_header != version["build_sha"]:
        raise SmokeCheckError("Build response header and /version build_sha disagree")
    if expected_sha and not _revisions_match(expected_sha, str(version["build_sha"])):
        raise SmokeCheckError(
            f"Deployed revision {version['build_sha']!r} does not match {expected_sha!r}"
        )

    with _request(base_url) as response:
        homepage = response.read().decode("utf-8", errors="replace")
        homepage_build = response.headers.get("X-TripBandhu-Build", "")

    if homepage_build != version["build_sha"]:
        raise SmokeCheckError("Homepage and /version were served by different builds")
    if "html2pdf" in homepage.casefold() or "jspdf" in homepage.casefold():
        raise SmokeCheckError("Legacy browser PDF exporter is still present in the homepage")

    asset_version = re.escape(str(version["asset_version"]))
    if not re.search(rf'/static/script\.js\?v={asset_version}(?:["\'])', homepage):
        raise SmokeCheckError("Homepage script asset does not use the reported asset version")
    if not re.search(rf'/static/style\.css\?v={asset_version}(?:["\'])', homepage):
        raise SmokeCheckError("Homepage stylesheet does not use the reported asset version")

    with _request(
        urljoin(base_url, "api/travel/download-pdf"),
        payload={"plan_id": SMOKE_PLAN_ID, "version": 1},
    ) as response:
        pdf = response.read()
        content_type = response.headers.get("Content-Type", "")
        disposition = response.headers.get("Content-Disposition", "")
        pdf_build = response.headers.get("X-TripBandhu-Build", "")

    if pdf_build != version["build_sha"]:
        raise SmokeCheckError("PDF and /version were served by different builds")
    if "application/pdf" not in content_type.casefold() or not pdf.startswith(b"%PDF-"):
        raise SmokeCheckError("Download endpoint did not return a valid PDF response")
    if "attachment" not in disposition.casefold():
        raise SmokeCheckError("PDF response is missing its attachment disposition")
    if b"jsPDF" in pdf:
        raise SmokeCheckError("Downloaded PDF still identifies the legacy jsPDF renderer")
    if len(pdf) < 2_000:
        raise SmokeCheckError("Downloaded PDF is unexpectedly small")

    return {
        "status": "ok",
        "base_url": base_url,
        "build_sha": version["build_sha"],
        "asset_version": version["asset_version"],
        "app_version": version["app_version"],
        "pdf_renderer": version["pdf_renderer"],
        "provider_config_version": version["provider_config_version"],
        "pdf_bytes": len(pdf),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="TripBandhu deployment URL")
    parser.add_argument("--expected-sha", default="", help="Expected Git commit SHA")
    args = parser.parse_args()
    try:
        result = verify_deployment(args.base_url, args.expected_sha)
    except (SmokeCheckError, ValueError, json.JSONDecodeError) as exc:
        print(f"DEPLOYMENT_SMOKE_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
