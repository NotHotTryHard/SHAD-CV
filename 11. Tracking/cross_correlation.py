import os

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_template

from detection import detection_cast, draw_detections, extract_detections
from tracker import Tracker


def gaussian(shape, x, y, dx, dy):
    """Return gaussian for tracking.

    shape: [width, height]
    x, y: gaussian center
    dx, dy: std by x and y axes

    return: numpy array (width x height) with gauss function, center (x, y) and std (dx, dy)
    """
    Y, X = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.exp(-((X - x) ** 2) / dx**2 - (Y - y) ** 2 / dy**2)


class CorrelationTracker(Tracker):
    """Generate detections and building tracklets."""

    def __init__(self, detection_rate=5, **kwargs):
        super().__init__(**kwargs)
        self.detection_rate = detection_rate  # Detection rate
        self.prev_frame = None  # Previous frame (used in cross correlation algorithm)

    def build_tracklet(self, frame):
        """Between CNN execution uses normalized cross-correlation algorithm (match_template)."""
        detections = []
        # Write code here
        # Apply rgb2gray to frame and previous frame
        frame_gray = rgb2gray(frame.copy())
        prev_frame_gray = rgb2gray(self.prev_frame.copy())

        # For every previous detection
        # Use match_template + gaussian to extract detection on current frame
        for label, xmin, ymin, xmax, ymax in self.detection_history[-1]:
            # Step 0: Extract prev_bbox from prev_frame
            #prev_bbox = prev_frame_gray[ymin: ymax, xmin + 1: xmax + 1]
            prev_bbox = prev_frame_gray[ymin: ymax, xmin: xmax]
            if prev_bbox.size == 0:
                continue

            # Step 1: Extract new_bbox from current frame with the same coordinates
            #new_bbox = frame_gray[ymin: ymax + 1, xmin: xmax + 1]
            new_bbox = frame_gray[ymin: ymax, xmin: xmax]
            if new_bbox.size == 0:
                continue

            # Step 2: Calc match_template between previous and new bbox
            # Use padding
            match = match_template(prev_bbox, new_bbox, pad_input=True)

            # Step 3: Then multiply matching by gauss function
            # Find argmax(matching * gauss)
            y = match.shape[0] // 2
            x = match.shape[1] // 2
            dx = match.shape[1]
            dy = match.shape[0]

            tmp = match * gaussian((dy, dx), x, y, dx, dy)
            max_y, max_x = np.unravel_index(np.argmax(tmp), tmp.shape)

            # Step 4: Append to detection list
            detections.append(
                np.array([
                    label,
                    xmin + (x - max_x),
                    ymin + (y - max_y),
                    xmax + (x - max_x),
                    ymax + (y - max_y)
                ])
            )

        return detection_cast(detections)
    

    def update_frame(self, frame):
        self.frame_id += 1
        if not self.frame_index:
            detections = self.init_tracklet(frame)
            self.save_detections(detections)
        elif self.frame_index % self.detection_rate == 0:
            detections = extract_detections(frame, labels=self.labels)
            detections = self.bind_tracklet(detections)
            self.save_detections(detections)
        else:
            detections = self.build_tracklet(frame)
        #if self.frame_id >= 500:
        #    io.imshow(draw_detections(frame, detections))
        #    io.show()
        self.detection_history.append(detections)
        self.prev_frame = frame
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections

    def update_frame(self, frame):
        if not self.frame_index:
            detections = self.init_tracklet(frame)
            self.save_detections(detections)
        elif self.frame_index % self.detection_rate == 0:
            detections = extract_detections(frame, labels=self.labels)
            detections = self.bind_tracklet(detections)
            self.save_detections(detections)
        else:
            detections = self.build_tracklet(frame)

        self.detection_history.append(detections)
        self.prev_frame = frame
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = CorrelationTracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
