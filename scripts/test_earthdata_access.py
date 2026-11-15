#!/usr/bin/env python3
"""Verify .netrc-based access to one protected IMERG Early granule."""

import earthaccess
from earthaccess.store import Store
import requests

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"

auth = earthaccess.login(strategy="netrc")
if not auth.authenticated:
    raise RuntimeError("Earthdata .netrc authentication failed")
store = Store(auth)

response = requests.get(CMR, params={
    "short_name": "GPM_3IMERGHHE",
    "version": "07",
    "temporal": "2021-07-25T00:00:00Z,2021-07-25T00:29:59Z",
    "page_size": 1,
}, timeout=60)
response.raise_for_status()
entry = response.json()["feed"]["entry"][0]
url = next(link["href"] for link in entry["links"]
           if link.get("rel", "").endswith("/data#")
           and link.get("href", "").endswith(".HDF5"))

store.set_requests_session(url)
session = store.get_requests_session()
with session.get(url, headers={"Range": "bytes=0-1023"}, timeout=120) as result:
    result.raise_for_status()
    payload = result.content
    print(f"Earthdata access OK: HTTP {result.status_code}, received {len(payload):,} bytes")
    print(f"Resolved URL: {result.url}")
