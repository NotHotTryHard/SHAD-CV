
def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    xlen1 = bbox1[2] - bbox1[0]
    xlen2 = bbox2[2] - bbox2[0]
    ylen1 = bbox1[3] - bbox1[1]
    ylen2 = bbox2[3] - bbox2[1]

    first_area = xlen1 * ylen1
    second_area = xlen2 * ylen2

    intersection_left = max(bbox1[0], bbox2[0])
    intersection_bottom = max(bbox1[1], bbox2[1])
    intersection_right = min(bbox1[2], bbox2[2])
    intersection_top = min(bbox1[3], bbox2[3])
    if intersection_right < intersection_left or intersection_top < intersection_bottom:
        return 0
    intersection_area = (intersection_right - intersection_left) * (intersection_top - intersection_bottom)

    return intersection_area / (first_area + second_area - intersection_area)


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        obj_dict = {}
        hyp_dict = {}
        for obj in frame_obj:
            obj_dict[obj[0]] = obj[1:]
        for hyp in frame_hyp:
            hyp_dict[hyp[0]] = hyp[1:]

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        for obj_id, hyp_id in matches.items():
            if obj_id in obj_dict.keys() and hyp_id in hyp_dict.keys() and iou_score(obj_dict[obj_id], hyp_dict[hyp_id]) > threshold:
                dist_sum += iou_score(obj_dict[obj_id], hyp_dict[hyp_id])
                match_count += 1
                obj_dict.pop(obj_id, None)
                hyp_dict.pop(hyp_id, None)


        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ids = []
        for obj_id in obj_dict.keys():
            for hyp_id in hyp_dict.keys():
                if iou_score(obj_dict[obj_id], hyp_dict[hyp_id]) > threshold:
                    ids.append((obj_id, hyp_id, iou_score(obj_dict[obj_id], hyp_dict[hyp_id])))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        ids = sorted(ids, key=lambda x: x[2], reverse=True)
        for obj_id, hyp_id, iou in ids:
            if obj_id in obj_dict.keys() and hyp_id in hyp_dict.keys():
                dist_sum += iou
                match_count += 1
                obj_dict.pop(obj_id, None)
                hyp_dict.pop(hyp_id, None)
                matches[obj_id] = hyp_id

        # Step 5: Update matches with current matched IDs
        

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    gt = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        obj_dict = {}
        hyp_dict = {}
        for obj in frame_obj:
            obj_dict[obj[0]] = obj[1:]
        for hyp in frame_hyp:
            hyp_dict[hyp[0]] = hyp[1:]

        gt += len(obj_dict.keys())

        for obj_id, hyp_id in matches.items():
            if obj_id in obj_dict.keys() and hyp_id in hyp_dict.keys() and iou_score(obj_dict[obj_id], hyp_dict[hyp_id]) > threshold:
                dist_sum += iou_score(obj_dict[obj_id], hyp_dict[hyp_id])
                match_count += 1
                obj_dict.pop(obj_id, None)
                hyp_dict.pop(hyp_id, None)

        ids = []
        for obj_id in obj_dict.keys():
            for hyp_id in hyp_dict.keys():
                if iou_score(obj_dict[obj_id], hyp_dict[hyp_id]) > threshold:
                    ids.append((obj_id, hyp_id, iou_score(obj_dict[obj_id], hyp_dict[hyp_id])))

        ids = sorted(ids, key=lambda x: x[2], reverse=True)

        for obj_id, hyp_id, iou in ids:
            if obj_id not in matches.keys() and hyp_id not in matches.values():
                dist_sum += iou
                match_count += 1
                matches[obj_id] = hyp_id
            elif obj_id in matches.keys() and matches[obj_id] != hyp_id:
                mismatch_error += 1
            if obj_id in obj_dict.keys():
                obj_dict.pop(obj_id, None)
            if hyp_id in hyp_dict.keys():
                hyp_dict.pop(hyp_id, None)

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        # Step 6: Update matches with current matched IDs
        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        false_positive += len(hyp_dict.keys())
        missed_count += len(obj_dict.keys())
    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count

    MOTA = 1 - (missed_count + false_positive + mismatch_error) / gt

    return MOTP, MOTA
