import os

import numpy as np

from detection import detection_cast, draw_detections, extract_detections
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
        frame = np.array(frame)
        detections = extract_detections(frame, min_confidence=0.6, labels=self.labels)

        labeled_detections = []
        for idx in range(detections.shape[0]):
            label = self.new_label()
            detections[idx][0] = label
            #labeled_detections.append(labeled_detection)

        return detections

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        detections = []
        found_idx = set()
        for i in range(min(self.lookup_tail_size, len(self.detection_history))):
            prev_detections = self.detection_history[-1-i]
            for detection in prev_detections:
                if detection[0] not in found_idx:
                    found_idx.add(detection[0])
                    detections.append(detection)

        return detection_cast(detections)


    def bind_tracklet(self, detections, iou_threshold=0.5):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        prev_detections = self.prev_detections

        # Write code here

        # Step 1: calc pairwise detection IOU
        detections = detections.copy()
        prev_detections = self.prev_detections

        # Create a list to store matched detections
        assigned_ids = set()

        # Initialize a dictionary to store the new IDs for current detections
        unmatched_indices = []

        # Step 1: Match current detections with previous detections
        # Step 2: sort IOU list
        # Step 3: fill detections[:, 0] with best match
        # One matching for each id
        for i, cur in enumerate(detections):
            best_iou = 0
            best_match_id = -1

            for j, prev in enumerate(prev_detections):
                iou = iou_score(cur[1:], prev[1:])
                if iou > best_iou and iou > iou_threshold and j not in assigned_ids:
                    best_iou = iou
                    best_match_id = prev[0]

            if best_match_id != -1:
                detections[i, 0] = best_match_id
                assigned_ids.add(best_match_id)
            else:
                unmatched_indices.append(i)
        # Step 4: assign new tracklet id to unmatched detections
        for idx in unmatched_indices:
            label = self.new_label()
            detections[idx, 0] = label

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
    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
