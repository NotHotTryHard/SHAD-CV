import numpy as np
import os

from moviepy.editor import VideoFileClip

from detection import detection_cast, extract_detections, draw_detections
from metrics import iou_score


class Tracker:
    """Generate detections and build tracklets."""
    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Write code here
        # Use extract_detections and new_label
        results = extract_detections(frame, labels=self.labels)
        for i in range(results.shape[0]):
            results[i][0] = self.new_label()
        return results

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        detections = []
        # Write code here
        ids = []
        for history in self.detection_history[-1:-self.lookup_tail_size:-1]:
            for detection in history:
                if detection[0] not in ids:
                    detections.append(detection)
                    ids.append(detection[0])

        return detection_cast(detections)

    def bind_tracklet(self, detections):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        prev_detections = self.prev_detections

        # Write code here

        # Step 1: calc pairwise detection IOU
        scores = []
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(prev_detections):
                scores.append((i, j, det2[0], iou_score(det1[1:], det2[1:])))

        # Step 2: sort IOU list
        scores = sorted(scores, key=lambda x: x[3], reverse=True)

        # Step 3: fill detections[:, 0] with best match
        # One matching for each id
        tracklet_ids = []
        detection_idxs = []
        for detection_idx, tracklet_idx, tracklet_id, iou in scores:
            if tracklet_id in tracklet_ids or detection_idx in detection_idxs:
                continue
            tracklet_ids.append(tracklet_id)
            detection_idxs.append(detection_idx)
            detections[detection_idx][0] = tracklet_id

        # Step 4: assign new tracklet id to unmatched detections
        for i in range(detections.shape[0]):
            if i not in detection_idxs:
                detections[i][0] = self.new_label()

        return detection_cast(detections)

    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, 'data', 'test.mp4'))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == '__main__':
    main()
