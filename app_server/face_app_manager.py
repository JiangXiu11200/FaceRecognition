import asyncio
import threading
from datetime import datetime
from queue import Queue

from face_detection import RunMode

from .config.adapter import ConfigAdapter
from .db.database import SessionLocal
from .db.models import SystemLogs
from .utils import post_log_to_server


class FaceAppManager:
    """Manager for FaceApp integration with FastAPI"""

    def __init__(self, connection_manager):
        from .connection_manager import ConnectionManager

        self.connection_manager: ConnectionManager = connection_manager
        self.config_adapter = ConfigAdapter()
        self.running = False
        self.face_app = None
        self.frame_queue = asyncio.Queue(maxsize=10)
        self.log_queue = asyncio.Queue(maxsize=100)
        self.detection_results_queue = Queue()

    def toggle_blink_detection(self):
        """Toggle blink detection in FaceApp"""
        if self.face_app:
            self.face_app.toggle_blink_detection()

    async def run(self):
        """Run face detection in async context"""
        self.running = True

        # Import FaceApp here to avoid circular import
        try:
            from face_detection import FaceApp

        except ImportError as e:
            print(f"Error importing FaceApp: {e}")
            return

        # Create FaceApp instance with our config adapter
        self.face_app = FaceApp(
            mode=RunMode.FASTAPI,
            config_source=self.config_adapter,
            frame_queue=self.frame_queue,
            log_queue=self.log_queue,
            external_detection_queue=self.detection_results_queue,
        )

        # Start face detection in a separate thread
        face_thread = threading.Thread(target=self.face_app.run)
        face_thread.daemon = True  # Allow thread to exit when main program exits
        face_thread.start()

        # Start async tasks for streaming and logging
        await asyncio.gather(self._stream_frames(), self._process_logs(), return_exceptions=True)

    async def stop(self):
        """Stop face detection"""
        self.running = False
        if self.face_app:
            self.face_app.stop()
            self.face_app = None

    async def _stream_frames(self):
        """Stream frames to WebSocket clients"""
        while self.running:
            try:
                # Get frame from queue
                frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=0.1)

                # Send to all connected clients
                await self.connection_manager.send_frame(frame_data)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error streaming frame: {e}")
                await asyncio.sleep(0.01)

    async def _process_logs(self):
        """Process detection logs"""
        while self.running:
            try:
                # Get log from queue
                log_data = await asyncio.wait_for(self.log_queue.get(), timeout=0.1)

                # Save to database
                await self._save_log_to_db(log_data)

                # Send to WebSocket clients
                await self.connection_manager.send_log(log_data)

                # Post to external server (if configured)
                if not self.config_adapter.system_config.debug:
                    asyncio.create_task(post_log_to_server(log_data))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing log: {e}")
                await asyncio.sleep(0.01)

    async def _save_log_to_db(self, log_data: dict):
        """Save detection log to database"""
        db = SessionLocal()
        try:
            db_log = SystemLogs(
                name=log_data.get("name", ""),
                group=log_data.get("group", ""),
                log_level=log_data.get("log_level", "INFO"),
                message=log_data.get("message", ""),
                timestamp=datetime.utcnow(),
            )
            db.add(db_log)
            db.commit()
        except Exception as e:
            print(f"Error saving log to database: {e}")
            db.rollback()
        finally:
            db.close()
