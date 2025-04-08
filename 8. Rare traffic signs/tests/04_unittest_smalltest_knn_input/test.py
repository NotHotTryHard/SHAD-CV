import csv
import json
import os

from rare_traffic_sign_solution import ModelWithHead, apply_classifier


def test_smalltest_knn():
    def read_csv(filename):
        res = {}
        with open(filename) as fhandle:
            reader = csv.DictReader(fhandle)
            for row in reader:
                res[row["filename"]] = row["class"]
        return res

    def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
        ok_cnt = 0
        all_cnt = 0
        for t, p in zip(y_true, y_pred):
            if cur_type == "all" or class_name_to_type[t] == cur_type:
                all_cnt += 1
                if t == p:
                    ok_cnt += 1
        return ok_cnt / max(1, all_cnt)

    def test_classifier(output, gt_file, classes_file):
        gt = read_csv(gt_file)
        y_pred = []
        y_true = []
        for k, v in output.items():
            y_pred.append(v)
            y_true.append(gt[k])

        with open(classes_file, "r") as fr:
            classes_info = json.load(fr)
        class_name_to_type = {k: v["type"] for k, v in classes_info.items()}

        total_acc = calc_metric(y_true, y_pred, "all", class_name_to_type)
        rare_recall = calc_metric(y_true, y_pred, "rare", class_name_to_type)
        freq_recall = calc_metric(y_true, y_pred, "freq", class_name_to_type)
        return total_acc, rare_recall, freq_recall

    ckpt_path = "improved_features_model.pth"
    head_path = "knn_model.bin"

    assert os.path.isfile(ckpt_path)
    assert os.path.isfile(head_path)

    model = ModelWithHead(n_neighbors=20)
    model.load_nn(ckpt_path)
    model.load_head(head_path)

    results = apply_classifier(model, "smalltest", "classes.json")
    results = {elem["filename"]: elem["class"] for elem in results}
    total_acc, rare_recall, freq_recall = test_classifier(
        results,
        "smalltest_annotations.csv",
        "classes.json",
    )
    assert freq_recall > 0.7
