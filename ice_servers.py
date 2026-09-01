import logging

import requests
import streamlit as st

logger = logging.getLogger(__name__)


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
