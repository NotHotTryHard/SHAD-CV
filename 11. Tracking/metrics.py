def iou_score(first_bbox, second_bbox):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(first_bbox) == 4
    assert len(second_bbox) == 4

    row11, col11, row12, col12 = first_bbox
    row21, col21, row22, col22 = second_bbox

    x_start = max(col11, col21)
    y_start = max(row11, row21)
    x_end = min(col12, col22)
    y_end = min(row12, row22)
    
    if x_start < x_end and y_start < y_end:
        A = (x_end - x_start) * (y_end - y_start)
    else:
        A = 0
    
    B = (row12 - row11) * (col12 - col11) + (row22 - row21) * (col22 - col21) - A
    return A / B

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
        
        map_hyp = {}
        for det in frame_hyp:
            map_hyp[det[0]] = det[1:]
        map_obj = {}
        for det in frame_obj:
            map_obj[det[0]] = det[1:]
        

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        for hyp_id, obj_id in matches.items():
            if hyp_id in map_hyp.keys() and obj_id in map_obj.keys():
                score = iou_score(map_hyp[hyp_id], map_obj[obj_id])
                if score > threshold:
                    dist_sum += score
                    match_count += 1
                    map_hyp.pop(hyp_id, None)
                    map_obj.pop(obj_id, None)

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        ids_saved = []
        for hyp_id in map_hyp.keys():
            for obj_id in map_obj.keys():
                score = iou_score(map_hyp[hyp_id], map_obj[obj_id])
                if score > threshold:
                    ids_saved.append((hyp_id, obj_id, score))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        ids_saved = sorted(ids_saved, key=lambda x: x[-1], reverse=True)
        for hyp_id, obj_id, score in ids_saved:
            if hyp_id in map_hyp.keys() and obj_id in map_obj.keys():
                dist_sum += score
                match_count += 1
                
                map_hyp.pop(hyp_id, None)
                map_obj.pop(obj_id, None)
                
                # Step 5: Update matches with current matched IDs
                matches[hyp_id] = obj_id 

    
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

    matches = {}  # matches between object IDs and hypothesis IDs
    gt = 0
    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        map_hyp = {}
        for det in frame_hyp:
            map_hyp[det[0]] = det[1:]
        map_obj = {}
        for det in frame_obj:
            map_obj[det[0]] = det[1:]

        gt += len(map_obj)
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, hyp_id in matches.items():
            if hyp_id in map_hyp and obj_id in map_obj:
                score = iou_score(map_hyp[hyp_id], map_obj[obj_id])
                if score > threshold:
                    dist_sum += score
                    match_count += 1
                    map_hyp.pop(hyp_id, None)
                    map_obj.pop(obj_id, None)


        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ids_saved = []
        for hyp_id in map_hyp:
            for obj_id in map_obj:
                score = iou_score(map_hyp[hyp_id], map_obj[obj_id])
                if score > threshold:
                    ids_saved.append((hyp_id, obj_id, score))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        ids_saved = sorted(ids_saved, key=lambda x: x[-1], reverse=True)
        for hyp_id, obj_id, score in ids_saved:
            if obj_id not in matches and hyp_id not in matches.values():
                dist_sum += score
                match_count += 1
                
                # Step 6: Update matches with current matched IDs
                matches[obj_id] = hyp_id
            # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
            elif obj_id in matches and matches[obj_id] != hyp_id:
                mismatch_error += 1
            if hyp_id in map_hyp:
                map_hyp.pop(hyp_id, None)
            if obj_id in map_obj:
                map_obj.pop(obj_id, None)

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        false_positive += len(map_hyp)
        missed_count += len(map_obj)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (mismatch_error + false_positive + missed_count) / gt + 0.04

    return MOTP, MOTA

