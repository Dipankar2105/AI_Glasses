import time

class VisionScheduler:
    def __init__(self, max_fps: float):
        self.max_fps = max_fps
        self.min_interval = 1.0 / max_fps if max_fps > 0 else 0
        self.last_process_time = -self.min_interval - 1.0
        self.frame_queue = []

    def push_frame(self, frame, timestamp_sec):
        # Latest frame behavior, drop old if accumulating
        if len(self.frame_queue) > 0:
            self.frame_queue.pop(0) # uncontrolled queue growth prevented
        self.frame_queue.append((frame, timestamp_sec))

    def get_next_frame(self, current_time_sec):
        if not self.frame_queue:
            return None, False
        frame, ts = self.frame_queue[0]
        if current_time_sec - self.last_process_time >= self.min_interval:
            self.frame_queue.pop(0)
            self.last_process_time = current_time_sec
            return frame, False
        else:
            # Drop frame due to throttling
            self.frame_queue.pop(0)
            return frame, True
