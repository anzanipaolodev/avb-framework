from datetime import datetime
import asyncio
from cores.avbcore import AVBCore
from agent import logger

class TestCore(AVBCore):
    def __init__(self):
        super().__init__("Test")
        self.last_log_time = None
        self.log_interval = 30 # 5 minutes in seconds

    def initialize(self):
        """Initialize the test core."""
        print(f"Initializing {self.core_name} core...")
        self.last_log_time = datetime.now()
        self.activate()  # Automatically activate the core on initialization

    def shutdown(self):
        """Cleanup and shutdown the test core."""
        print(f"Shutting down {self.core_name} core...")
        self.deactivate()

    def _tick(self):
        """Log a message every 5 minutes."""
        current_time = datetime.now()
        if self.last_log_time is None or (current_time - self.last_log_time).total_seconds() >= self.log_interval:
            logger.async_log(f"Test Core: Logging test message at {current_time}")
            self.last_log_time = current_time 
