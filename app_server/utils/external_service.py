import logging

import aiohttp

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
    if not server_url:
        # Get from environment variable or config
        import os

        server_url = os.getenv("EXTERNAL_LOG_SERVER_URL")

    if not server_url:
        logger.warning("No external log server URL configured")
        return

    try:
        async with aiohttp.ClientSession() as session:
            # FIXME: 需要修正 post 過去的資料格式
            async with session.post(f"{server_url}", json=log_data, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    logger.info("Successfully posted log to external server")
                else:
                    logger.error(f"Failed to post log: {response.status}")
    except Exception as e:
        logger.error(f"Error posting log to external server: {e}")
