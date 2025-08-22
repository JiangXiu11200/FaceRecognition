import asyncio
import threading

from .config.adapter import ConfigAdapter


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
        )

        face_thread = threading.Thread(target=self.video_stream.run)
        face_thread.daemon = True
        face_thread.start()

        await asyncio.gather(self._stream_frames(), return_exceptions=True)

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
