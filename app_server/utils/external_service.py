import logging
import os
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

CURRENT_FILE = Path(__file__).resolve()
ENV_PATH = CURRENT_FILE.parents[2]
load_dotenv(ENV_PATH / ".env")

ACTIVITY_LOGS_URL = os.getenv("EXTERNAL_ACTIVITY_LOGS_SERVER_URL")
ALARM_LOGS_URL = os.getenv("EXTERNAL_ALARM_LOGS_SERVER_URL")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def post_log_to_server(log_data: dict, server_url: str = None):
    """
    Post detection log to external server

    Args:
        log_data: Detection log data
        server_url: External server URL (from environment variable or config)
    """
    server_url = ACTIVITY_LOGS_URL if log_data.get("detection_results") else ALARM_LOGS_URL

    if not server_url:
        logger.warning("No external log server URL configured")
        return

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{server_url}", json=log_data, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    logger.info("Successfully posted log to external server")
                else:
                    try:
                        content_type = response.headers.get("Content-Type", "")
                        if "application/json" in content_type:
                            data = await response.json()
                        else:
                            data = await response.text()
                        logger.info(f"Response from server:{response.status}: {data}")
                    except Exception as parse_error:
                        print(f"Error parsing response: {parse_error}")
    except Exception as e:
        logger.error(f"Error posting log to external server: {e}")
