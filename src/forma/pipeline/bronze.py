"""Bronze: download every source in sources.yaml exactly as published, with checksums and provenance.

Files land in data/raw/<source id>/. A re-run skips anything already downloaded unless `refresh`.
Each download is recorded in bronze.files with its URL, license, size, SHA-256 and retrieval time.
"""

from __future__ import annotations

import hashlib
import json
import string
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ..config import Settings

UA = {"User-Agent": "Forma-pipeline/0.1 (educational fitness coach; github.com/Manvith-d/forma)"}
EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
LICENSE_NAMES = {"cc by": "cc-by", "cc0": "cc0"}


class LicenseError(RuntimeError):
    pass


def _get(client: httpx.Client, url: str, **params) -> httpx.Response:
    for attempt in range(5):  # gov servers drop connections, rate-limit (429) or briefly bot-block cloud IPs (403)
        try:
            r = client.get(url, params=params or None)
            if r.status_code < 500 and r.status_code not in (403, 429):
                r.raise_for_status()
                return r
        except httpx.TransportError:
            if attempt == 4:
                raise
        time.sleep(2 * 2 ** attempt)
    r.raise_for_status()
    return r


def _save(path: Path, content: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def fetch_all(s: Settings, manifest: dict, refresh: bool = False) -> list[dict]:
    """Download every document and dataset. Returns one provenance row per file."""
    rows: list[dict] = []
    raw = s.data / "raw"
    with httpx.Client(headers=UA, timeout=120, follow_redirects=True) as client:
        def keep(source_id: str, url: str, license: str, path: Path, fetch) -> None:
            if path.exists() and not refresh:
                content = path.read_bytes()
                rows.append({"source_id": source_id, "url": url, "license": license, "path": str(path),
                             "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                             "fetched_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()})
                return
            rows.append({"source_id": source_id, "url": url, "license": license,
                         "fetched_at": datetime.now(UTC).isoformat(), **_save(path, fetch())})

        for doc in manifest["documents"]:
            d = raw / doc["id"]
            if doc["kind"] == "europepmc":
                meta_path = d / "meta.json"
                keep(doc["id"], f"{EUROPE_PMC}/search?query=PMCID:{doc['pmcid']}", doc["license"], meta_path,
                     lambda doc=doc: _get(client, f"{EUROPE_PMC}/search", query=f"PMCID:{doc['pmcid']}",
                                          format="json", resultType="core").content)
                meta = json.loads(meta_path.read_text())["resultList"]["result"][0]
                found = LICENSE_NAMES.get((meta.get("license") or "").lower())
                if found != doc["license"]:  # trust the publisher's metadata, not the manifest
                    raise LicenseError(f"{doc['id']}: Europe PMC reports license {meta.get('license')!r}")
                keep(doc["id"], f"{EUROPE_PMC}/{doc['pmcid']}/fullTextXML", doc["license"], d / "fulltext.xml",
                     lambda doc=doc: _get(client, f"{EUROPE_PMC}/{doc['pmcid']}/fullTextXML").content)
            else:
                name = "document.pdf" if doc["kind"] == "pdf" else "page.html"
                keep(doc["id"], doc["url"], doc["license"], d / name, lambda doc=doc: _get(client, doc["url"]).content)

        for ds in manifest["datasets"]:
            d = raw / ds["id"]
            if ds["id"] == "openfda-labels":
                for generics in manifest["medications"].values():
                    for g in generics:
                        keep(ds["id"], f"{ds['url']}?count=openfda.brand_name.exact&search=generic:\"{g}\"", ds["license"],
                             d / "brands" / f"{g.replace(' ', '_')}.json",
                             lambda g=g, url=ds["url"]: _get(client, url, search=f'openfda.generic_name:"{g}"',
                                                              count="openfda.brand_name.exact", limit=100).content)
                        keep(ds["id"], f"{ds['url']}?search=openfda.generic_name:\"{g}\"", ds["license"],
                             d / f"{g.replace(' ', '_')}.json",
                             lambda g=g, url=ds["url"]: _get(client, url, search=f'openfda.generic_name:"{g}"',
                                                              limit=5).content)
            elif ds["id"] == "themealdb":
                for letter in string.ascii_lowercase:
                    keep(ds["id"], f"{ds['url']}?f={letter}", ds["license"], d / f"{letter}.json",
                         lambda letter=letter, url=ds["url"]: _get(client, url, f=letter).content)
            else:
                name = ds["url"].rsplit("/", 1)[-1]
                keep(ds["id"], ds["url"], ds["license"], d / name, lambda ds=ds: _get(client, ds["url"]).content)
    return rows
