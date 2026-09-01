"""
ICE server (STUN/TURN) configuration for WebRTC.

Streamlit Community Cloud's network does not complete a WebRTC peer
connection with STUN alone -- a TURN relay is required (confirmed via
streamlit-webrtc's own maintainer). This module provides two free
options, no paid service, no VPS:

1. A free account + API key at metered.ca (Open Relay Project), which
   fetches time-bound credentials via their REST API. More reliable,
   requires a one-time free signup (email-based, not phone verification).
2. Fallback: the static, zero-signup public credentials Open Relay
   Project publishes for anyone to use. Convenient, but explicitly
   flagged by streamlit-webrtc's maintainer as unreliable -- it can go
   down or get overloaded since it's a shared public resource with no
   account behind it.

If you sign up for a free Metered account, set these two Streamlit
secrets (Settings -> Secrets on Streamlit Cloud, or .streamlit/secrets.toml
locally):

    METERED_APP_NAME = "your-app-name"   # the subdomain in your dashboard
    METERED_API_KEY = "your-api-key"

Without those set, this automatically falls back to the static
credentials -- the app will still run, just less reliably.
"""

import logging

import requests
import streamlit as st

logger = logging.getLogger(__name__)

# Static, zero-signup fallback credentials (Open Relay Project's public
# shared TURN server). No account needed, but shared/public and known to
# be unreliable under load -- see module docstring.
_FALLBACK_ICE_SERVERS = [
    {"urls": "stun:stun.relay.metered.ca:80"},
    {
        "urls": "turn:standard.relay.metered.ca:80",
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
    {
        "urls": "turn:standard.relay.metered.ca:443",
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
]


@st.cache_data(ttl=1800)  # Metered credentials are time-bound; refresh periodically
def get_ice_servers():
    """Returns a list of ICE server dicts for RTCConfiguration. Tries a
    Metered API key from st.secrets first; falls back to the static
    public Open Relay credentials if no key is configured or the API
    call fails."""
    try:
        app_name = st.secrets["METERED_APP_NAME"]
        api_key = st.secrets["METERED_API_KEY"]
    except (KeyError, FileNotFoundError):
        logger.info(
            "No METERED_APP_NAME/METERED_API_KEY in st.secrets -- using "
            "the static public Open Relay fallback credentials."
        )
        return _FALLBACK_ICE_SERVERS

    url = f"https://{app_name}.metered.live/api/v1/turn/credentials"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=5)
        resp.raise_for_status()
        ice_servers = resp.json()
        if not ice_servers:
            raise ValueError("Metered API returned an empty ice server list")
        return ice_servers
    except Exception as e:
        logger.warning(
            f"Failed to fetch Metered TURN credentials ({e}) -- falling "
            "back to the static public Open Relay credentials."
        )
        return _FALLBACK_ICE_SERVERS
