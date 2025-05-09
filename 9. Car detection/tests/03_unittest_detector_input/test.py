from common import calc_detector_auc
from os.path import dirname, join


def test_detector():
    test_dir = dirname(__file__)
    img_dir = join(test_dir, 'test_imgs')
    gt_path = join(test_dir, 'true_detections.json')

    score = calc_detector_auc(img_dir, gt_path)
    print(score)
    if score < 0.2:
        raise ValueError("Too small detector AUC score before NMS: " + str(score))
