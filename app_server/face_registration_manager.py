import asyncio
import threading

from .config.adapter import ConfigAdapter
from .utils.external_service import post_log_to_server


class FaceRegistrationManager:
    """Manager for FaceApp integration with FastAPI"""

    def __init__(self, connection_manager):
        from .connection_manager import ConnectionManager

        self.connection_manager: ConnectionManager = connection_manager
        self.config_adapter = ConfigAdapter()
        self.running = False
        self.video_stream = None
        self.frame_queue = asyncio.Queue(maxsize=10)
        self.log_queue = asyncio.Queue(maxsize=100)

    async def run(self):
        """Run face detection in async context"""
        self.running = True

        try:
            from package.video_stream import VideoStream

        except ImportError as e:
            print(f"Error importing FaceApp: {e}")
            return

        self.video_stream = VideoStream(
            config_source=self.config_adapter,
            frame_queue=self.frame_queue,
            log_queue=self.log_queue,
        )

        face_thread = threading.Thread(target=self.video_stream.run)
        face_thread.daemon = True
        face_thread.start()

        await asyncio.gather(self._stream_frames(), self._log_handler(), return_exceptions=True)

    async def stop(self):
        """Stop face detection"""
        self.running = False
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None

    async def _stream_frames(self):
        """Stream frames to WebSocket clients"""
        while self.running:
            try:
                frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=0.1)
                await self.connection_manager.send_frame(frame_data)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error streaming frame: {e}")
                await asyncio.sleep(0.01)

    async def _log_handler(self):
        """Process detection logs"""
        while self.running:
            try:
                log_data = await asyncio.wait_for(self.log_queue.get(), timeout=0.1)
                await self.connection_manager.send_log(log_data)

                if not self.config_adapter.system_config.debug:
                    asyncio.create_task(post_log_to_server(log_data))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing log: {e}")
                await asyncio.sleep(0.01)
