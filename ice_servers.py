import logging
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# Public STUN server used only as a fallback.
# We intentionally do NOT use hard-coded TURN credentials because
# invalid/expired TURN credentials can cause the 401 error seen in aiortc.
_FALLBACK_ICE_SERVERS = [
    {
        "urls": "stun:stun.l.google.com:19302",
    },
]


@st.cache_data(ttl=900)
def get_ice_servers():
    """Fetch fresh Metered TURN credentials for WebRTC.

    Streamlit secrets must contain:
        METERED_APP_NAME = "your-mettered-app-name"
        METERED_API_KEY  = "your-mettered-api-key"

    Returns a list suitable for RTCConfiguration(iceServers=[...]).
    """

    # Read credentials from Streamlit secrets.
    try:
        app_name = str(st.secrets["METERED_APP_NAME"]).strip()
        api_key = str(st.secrets["METERED_API_KEY"]).strip()
    except (KeyError, FileNotFoundError):
        logger.warning(
            "METERED_APP_NAME or METERED_API_KEY is missing from "
            "Streamlit secrets. Using STUN-only configuration."
        )
        return _FALLBACK_ICE_SERVERS

    if not app_name or not api_key:
        logger.warning(
            "METERED_APP_NAME or METERED_API_KEY is empty. "
            "Using STUN-only configuration."
        )
        return _FALLBACK_ICE_SERVERS

    # Metered's credential endpoint.
    url = f"https://{app_name}.metered.live/api/v1/turn/credentials"

    try:
        response = requests.get(
            url,
            params={"apiKey": api_key},
            timeout=10,
        )

        # Log the HTTP status without exposing the API key.
        logger.info(
            "Metered TURN credential request returned HTTP %s",
            response.status_code,
        )

        response.raise_for_status()

        ice_servers = response.json()

        if not isinstance(ice_servers, list) or not ice_servers:
            raise ValueError(
                "Metered API returned an empty or invalid ICE server list."
            )

        # Validate the returned structure before giving it to aiortc.
        valid_servers = []

        for server in ice_servers:
            if not isinstance(server, dict):
                continue

            urls = server.get("urls")
            if not urls:
                continue

            cleaned = {"urls": urls}

            # Keep credentials only when Metered supplied them.
            if server.get("username") is not None:
                cleaned["username"] = server["username"]

            if server.get("credential") is not None:
                cleaned["credential"] = server["credential"]

            valid_servers.append(cleaned)

        if not valid_servers:
            raise ValueError(
                "Metered API returned no usable ICE servers."
            )

        logger.info(
            "Successfully loaded %d ICE server entries from Metered.",
            len(valid_servers),
        )

        return valid_servers

    except requests.exceptions.HTTPError as e:
        # Do not fall back to fake/static TURN credentials.
        # A Metered 401/403 should be visible because using invalid TURN
        # credentials is what causes the aiortc 401 error.
        status = (
            e.response.status_code
            if e.response is not None
            else "unknown"
        )

        logger.error(
            "Metered TURN credential request failed with HTTP %s. "
            "Check METERED_APP_NAME and METERED_API_KEY in Streamlit "
            "secrets.",
            status,
        )

        # STUN-only fallback is safer than invalid TURN credentials.
        return _FALLBACK_ICE_SERVERS

    except requests.exceptions.RequestException as e:
        logger.error(
            "Could not reach Metered TURN credential service: %s",
            e,
        )
        return _FALLBACK_ICE_SERVERS

    except (ValueError, TypeError) as e:
        logger.error(
            "Invalid response from Metered TURN credential service: %s",
            e,
        )
        return _FALLBACK_ICE_SERVERS

    except Exception as e:
        logger.exception(
            "Unexpected error while obtaining Metered ICE servers: %s",
            e,
        )
        return _FALLBACK_ICE_SERVERS
