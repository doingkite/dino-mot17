from pathlib import Path
import cv2
import numpy as np
from hpatches_spatial_eval import discover_pairs, prepare_image, load_ground_truth, geometry_error

root = Path("data/hpatches/hpatches-sequences-release")
pairs = discover_pairs(root, categories=("i",))
errors = []

for _, sequence, target, source_path, target_path, gt_path in pairs:
    source = cv2.imread(str(source_path))
    target_image = cv2.imread(str(target_path))
    _, source_transform, source_valid = prepare_image(source)
    _, target_transform, target_valid = prepare_image(target_image)
    truth = load_ground_truth(gt_path, source_transform, target_transform)
    error = geometry_error(np.eye(3), truth, source_valid, target_valid)
    errors.append(error)

valid = [e for e in errors if e is not None]
print("평가 쌍:", len(errors))
print("정지 기준선 오차 중앙값:", float(np.median(valid)))
print("5px 이내:", sum(e is not None and e <= 5 for e in errors), "/", len(errors))
print("10px 이내:", sum(e is not None and e <= 10 for e in errors), "/", len(errors))
