"""
Blink detection module for face recognition system.
"""

from typing import Optional

import cv2
import numpy as np

import package.calculation as calculation
from package import config


class BlinkDetector:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.eyes_blink: list[list] = [[], []]
        self.blink_state: bool = False
        self.blink_count: int = 0
        self.left_median: int = 1
        self.right_median: int = 1
        self.average_brightness: np.float64 = 0
        self.threshold_value: int = 0

    def set_enabled(self, enabled: bool):
        """Set blink detection enabled state."""
        self.enabled = enabled
        if not enabled:
            self.reset()

    def reset(self):
        """Reset blink detection state."""
        self.eyes_blink = [[], []]
        self.blink_count = 0
        self.average_brightness = 0
        self.blink_state = False

    def update_brightness(self, face_roi: np.ndarray, brightness_threshold: int, brightness_values: list[int]) -> int:
        """
        Update brightness threshold based on face ROI.

        Parameters:
            face_roi: Face region of interest
            brightness_threshold: brightness threshold for detection
            brightness_values: brightness values list [high brightness value, low brightness value]

        Returns:
            threshold_value: Updated brightness threshold value
        """
        if not self.enabled:
            return 0

        hsv_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        _, _, value = cv2.split(hsv_image)
        self.average_brightness = np.mean(value)

        if self.average_brightness > brightness_threshold:
            self.threshold_value = brightness_values[0]
        else:
            self.threshold_value = brightness_values[1]

        config.logger.debug(f"face bounding box average brightness: {self.average_brightness}")
        return self.threshold_value

    def process_eyes(self, left_eye_gary: Optional[np.ndarray], right_eye_gary: Optional[np.ndarray]) -> bool:
        """
        Handle eye images to detect blink state.

        Parameters:
            left_eye_gary: left eye grayscale image
            right_eye_gary: right eye grayscale image

        Returns:
            blink_state: True if blink detected, False otherwise
        """
        if not self.enabled:
            return False

        if left_eye_gary is not None or right_eye_gary is not None:
            self.eyes_blink[0].append((left_eye_gary == 0).sum())
            self.eyes_blink[1].append((right_eye_gary == 0).sum())

        if len(self.eyes_blink[0]) > 15 and len(self.eyes_blink[1]) > 15:
            self.blink_state, self.left_median, self.right_median = calculation.Calculation.blink_detect(
                self.eyes_blink, self.blink_count, self.left_median, self.right_median
            )

        return self.blink_state

    def increment_count(self):
        """Increment blink count."""
        if self.enabled:
            self.blink_count += 1

    def should_update_brightness(self, interval: int = 5) -> bool:
        """
        Check if the blink detector should update brightness.

        Parameters:
            interval: Interval for updating brightness

        Returns:
            bool: True if update is needed, False otherwise
        """
        return self.enabled and self.average_brightness == 0 and self.blink_count % interval == 0
