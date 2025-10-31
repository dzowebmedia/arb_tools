#!/usr/bin/env python3
import os
import json
import urllib.parse
from pathlib import Path

from dotenv import load_dotenv
from bingads.authorization import OAuthWebAuthCodeGrant

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CLIENT_ID = os.getenv("MSADS_CLIENT_ID") or os.getenv("CLIENT_ID") or ""
CLIENT_SECRET = os.getenv("MSADS_CLIENT_SECRET") or os.getenv("CLIENT_SECRET") or ""
TENANT = os.getenv("MSADS_AUTH_TENANT") or "common"
REDIRECT_URI = os.getenv("MSADS_REDIRECT_URI") or "https://localhost:8000/auth/callback"

SCOPE = "openid offline_access https://ads.microsoft.com/msads.manage"
TOKEN_FILE = BASE_DIR / "msads_tokens.json"

if not CLIENT_ID:
    raise SystemExit("Missing MSADS_CLIENT_ID or CLIENT_ID in .env")
if not CLIENT_SECRET:
    raise SystemExit("Missing MSADS_CLIENT_SECRET or CLIENT_SECRET in .env")

auth_base = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/authorize"
params = {
    "client_id": CLIENT_ID,
    "response_type": "code",
    "redirect_uri": REDIRECT_URI,
    "response_mode": "query",
    "scope": SCOPE,
    "state": "12345",
    "prompt": "login",
}
auth_url = auth_base + "?" + urllib.parse.urlencode(params)

print("Open this URL in your browser:")
print(auth_url)
print()
redirect = input("Paste the FULL redirect URL here: ").strip()

oauth = OAuthWebAuthCodeGrant(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirection_uri=REDIRECT_URI,
)
oauth.request_oauth_tokens_by_response_uri(redirect)
tokens = oauth.oauth_tokens

out = {
    "access_token": getattr(tokens, "access_token", None),
    "refresh_token": getattr(tokens, "refresh_token", None),
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "redirect_uri": REDIRECT_URI,
    "scope": SCOPE,
    "tenant": TENANT,
}

TOKEN_FILE.write_text(json.dumps(out, indent=2))
print(f"Saved tokens to {TOKEN_FILE}")