"""Compare frozen DINOv3 patch matching with SIFT (and optional hybrids) on HPatches.

The supplied homography is loaded only after every method has estimated its own
geometry. Matching and RANSAC run in a common letterboxed canvas (default 448px);
metrics are reported both in that canvas (v1, unchanged) and at ORIGINAL image
resolution (corner error and mean matching accuracy, comparable to literature).
This is planar two-view registration, not metric 3D reconstruction.

v2 additions (all off by default, so v1 commands reproduce v1 numbers):
  --dino-ransac-threshold   separate RANSAC threshold for patch-grid matches
  --refine soft|quadratic   sub-patch localisation from the 3x3 similarity neighbourhood
  --target-perturb          shift / random homography applied to the target image so that
                            identity-GT sequences (HPatches i_*) cannot be solved by grid
                            coincidence; GT is composed with the same perturbation
  methods DINOguidedSIFT, SIFTdinoFilter, LoFTR, LightGlue
  --visualize-pairs v_bark:3 ...
"""

import argparse
import csv
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

MMA_THRESHOLDS = tuple(range(1, 11))
HACC_THRESHOLDS = (1, 3, 5, 10)
ALL_METHODS = ("DINOv3", "SIFT", "DINOguidedSIFT", "SIFTdinoFilter", "LoFTR", "LightGlue")
DINO_FEATURE_METHODS = ("DINOv3", "DINOguidedSIFT", "SIFTdinoFilter")


def prepare_image(bgr, size=448, mask=None):
    """Letterbox without stretching; return canvas, original->canvas matrix, mask.

    ``mask`` (optional, HxW uint8) marks valid original pixels, e.g. after a synthetic warp.
    """
    height, width = bgr.shape[:2]
    if not height or not width:
        raise ValueError("Empty image")
    scale = min(size / width, size / height)
    new_width = min(size, max(1, round(width * scale)))
    new_height = min(size, max(1, round(height * scale)))
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    resized = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + new_height, left:left + new_width] = resized
    valid = np.zeros((size, size), dtype=np.uint8)
    if mask is None:
        valid[top:top + new_height, left:left + new_width] = 1
    else:
        valid[top:top + new_height, left:left + new_width] = cv2.resize(
            (mask > 0).astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    transform = np.array([[new_width / width, 0, left],
                          [0, new_height / height, top], [0, 0, 1]], dtype=np.float64)
    return canvas, transform, valid


def discover_pairs(root, categories=("v",), max_sequences=None):
    """Yield deterministic full-image HPatches pairs (reference image 1)."""
    pairs = []
    for category in categories:
        folders = sorted(p for p in root.glob(f"{category}_*") if p.is_dir())
        if max_sequences is not None:
            folders = folders[:max_sequences]
        for folder in folders:
            for number in range(2, 7):
                images = [find_image(folder, n) for n in (1, number)]
                homography = folder / f"H_1_{number}"
                if images[0] is None or images[1] is None or not homography.is_file():
                    raise FileNotFoundError(f"Incomplete HPatches full sequence: {folder}, pair 1:{number}")
                pairs.append((category, folder.name, number, *images, homography))
    if not pairs:
        raise FileNotFoundError(f"No complete HPatches full sequences under {root}; expected v_*/1.ppm, 2.ppm, H_1_2")
    return pairs


def find_image(folder, number):
    for suffix in ("ppm", "png", "jpg", "jpeg"):
        path = folder / f"{number}.{suffix}"
        if path.is_file():
            return path
    return None


def load_homography(path):
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError(f"Invalid homography: {path}")
    return matrix


def load_ground_truth(path, source_transform, target_transform, target_perturbation=None):
    """GT in canvas coordinates. H_1_X maps original image 1 pixels into original image X pixels.

    ``target_perturbation`` (3x3, original target pixels) is the synthetic warp applied to
    image X before matching; the effective GT becomes P @ H.
    """
    matrix = load_homography(path)
    if target_perturbation is not None:
        matrix = target_perturbation @ matrix
    return target_transform @ matrix @ np.linalg.inv(source_transform)


def project(matrix, xy):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((xy, np.ones(len(xy)))) @ matrix.T
    denom = homogeneous[:, 2]
    output = np.full((len(xy), 2), np.nan, dtype=np.float64)
    good = np.abs(denom) > 1e-9
    output[good] = homogeneous[good, :2] / denom[good, None]
    return output


def stable_rng(seed, *keys):
    text = "|".join(str(k) for k in (seed, *keys))
    return np.random.default_rng(zlib.crc32(text.encode("utf-8")))


def make_perturbation(shape, mode, magnitude, rng):
    """3x3 warp in original target pixels: 'shift' = random sub-patch translation,
    'homography' = each corner displaced by up to ``magnitude`` px."""
    if mode in (None, "none"):
        return None
    height, width = shape[:2]
    if mode == "shift":
        dx, dy = rng.uniform(-magnitude, magnitude, 2)
        # Avoid an exact multiple of the 16px patch at canvas scale by construction: draw floats.
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
    if mode == "homography":
        corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        moved = corners + rng.uniform(-magnitude, magnitude, corners.shape).astype(np.float32)
        return cv2.getPerspectiveTransform(corners, moved).astype(np.float64)
    raise ValueError(f"Unknown perturbation {mode}")


def apply_perturbation(image, matrix):
    height, width = image.shape[:2]
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((height, width), np.uint8), matrix, (width, height),
                               flags=cv2.INTER_NEAREST)
    return warped, mask


def geometry_error(predicted, ground_truth, source_valid, target_valid):
    """Median reprojection discrepancy on a fixed grid of true overlap pixels (canvas px)."""
    if predicted is None:
        return None
    size = source_valid.shape[0]
    values = np.linspace(0, size - 1, 17)
    grid = np.array([(x, y) for y in values for x in values], dtype=np.float64)
    source_inside = source_valid[np.clip(np.rint(grid[:, 1]).astype(int), 0, size - 1),
                                 np.clip(np.rint(grid[:, 0]).astype(int), 0, size - 1)] > 0
    grid = grid[source_inside]
    truth = project(ground_truth, grid)
    target_idx = np.rint(np.nan_to_num(truth, nan=-1e6, posinf=-1e6, neginf=-1e6)).astype(np.int64)
    inside = np.isfinite(truth).all(axis=1) & (target_idx >= 0).all(axis=1) & (target_idx < size).all(axis=1)
    inside[inside] &= target_valid[target_idx[inside, 1], target_idx[inside, 0]] > 0
    if not inside.any():
        raise ValueError("Ground-truth homography has no visible overlap")
    estimate = project(predicted, grid[inside])
    if not np.isfinite(estimate).all():
        return None
    return float(np.median(np.linalg.norm(estimate - truth[inside], axis=1)))


def to_original_homography(canvas_h, source_transform, target_transform):
    if canvas_h is None:
        return None
    return np.linalg.inv(target_transform) @ canvas_h @ source_transform


def corner_error(predicted_original, truth_original, source_shape):
    """Mean distance of the 4 source-image corners (original target pixels), SuperPoint-style."""
    if predicted_original is None:
        return None
    height, width = source_shape[:2]
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float64)
    estimate, truth = project(predicted_original, corners), project(truth_original, corners)
    if not (np.isfinite(estimate).all() and np.isfinite(truth).all()):
        return None
    return float(np.mean(np.linalg.norm(estimate - truth, axis=1)))


def correspondence_errors(source_xy, target_xy, ground_truth, target_valid):
    """Per-match error; matches whose true target is invisible get +inf."""
    count = len(source_xy)
    if count != len(target_xy):
        raise ValueError("Source and target match counts differ")
    errors = np.full(count, np.inf, dtype=np.float64)
    if not count:
        return errors, np.zeros(0, bool)
    truth_xy = project(ground_truth, source_xy)
    height, width = target_valid.shape
    finite = np.isfinite(truth_xy).all(axis=1)
    xy_int = np.rint(np.nan_to_num(truth_xy, nan=-1e6, posinf=-1e6,
                                  neginf=-1e6)).astype(np.int64)
    visible = finite & (xy_int[:, 0] >= 0) & (xy_int[:, 0] < width) & \
        (xy_int[:, 1] >= 0) & (xy_int[:, 1] < height)
    visible[visible] &= target_valid[xy_int[visible, 1], xy_int[visible, 0]] > 0
    errors[visible] = np.linalg.norm(np.asarray(target_xy, np.float64)[visible] - truth_xy[visible], axis=1)
    return errors, visible


def match_quality(source_xy, target_xy, ground_truth, target_valid):
    """Score predicted correspondences using GT only after matching is complete.

    Precision uses every proposed match as its denominator. A source point whose
    true target lies outside the visible target image is an incorrect proposal.
    This is correspondence precision, not recall or registration success.
    """
    count = len(source_xy)
    if count != len(target_xy):
        raise ValueError("Source and target match counts differ")
    if not count:
        return {"gt_visible_matches": 0, "correct_matches_5px": 0,
                "correct_matches_10px": 0, "correct_matches_16px": 0,
                "match_precision_5px": None, "match_precision_10px": None,
                "match_precision_16px": None}
    errors, visible = correspondence_errors(source_xy, target_xy, ground_truth, target_valid)
    result = {"gt_visible_matches": int(np.count_nonzero(visible))}
    for threshold in (5, 10, 16):
        correct = int(np.count_nonzero(errors <= threshold))
        result[f"correct_matches_{threshold}px"] = correct
        result[f"match_precision_{threshold}px"] = correct / count
    return result


def original_match_quality(source_xy, target_xy, source_transform, target_transform,
                           truth_original, target_valid_original):
    """Mean matching accuracy at ORIGINAL resolution for thresholds 1..10 px (correct counts)."""
    if len(source_xy) == 0:
        return {f"correct_orig_{t}px": 0 for t in MMA_THRESHOLDS}
    source_o = project(np.linalg.inv(source_transform), source_xy)
    target_o = project(np.linalg.inv(target_transform), target_xy)
    errors, _ = correspondence_errors(source_o, target_o, truth_original, target_valid_original)
    return {f"correct_orig_{t}px": int(np.count_nonzero(errors <= t)) for t in MMA_THRESHOLDS}


def select_spread_matches(source, target, scores, size, limit=400):
    """Limit matches per source cell to avoid a single textured region dominating."""
    if len(scores) == 0:
        return source, target, scores
    buckets = {}
    selected = []
    for index in np.argsort(-scores):
        x, y = source[index]
        key = (min(7, int(x * 8 / size)), min(7, int(y * 8 / size)))
        if buckets.get(key, 0) < 8:
            selected.append(index)
            buckets[key] = buckets.get(key, 0) + 1
        if len(selected) >= limit:
            break
    return source[selected], target[selected], scores[selected]


def mutual_patch_matches(source_features, source_xy, target_features, target_xy, size,
                         refine=None, target_grid=None, patch_size=16, temperature=0.05):
    """Mutual nearest neighbours between patch tokens.

    If ``refine`` is set, ``target_grid`` = (grid_features[g,g,C], grid_valid[g,g]) is used to
    move each target point inside its patch using the 3x3 similarity neighbourhood.
    """
    if len(source_features) == 0 or len(target_features) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty(0, np.float32)
    similarities = source_features @ target_features.T
    forward = similarities.argmax(axis=1)
    reverse = similarities.argmax(axis=0)
    ids = np.arange(len(forward))
    keep = ids[reverse[forward] == ids]
    matched_target = target_xy[forward[keep]].astype(np.float32)
    if refine and refine != "none":
        if target_grid is None:
            raise ValueError("Sub-patch refinement needs the full target feature grid")
        matched_target = refine_targets(source_features[keep], matched_target, target_grid,
                                        patch_size, refine, temperature)
    return select_spread_matches(source_xy[keep], matched_target,
                                 similarities[keep, forward[keep]], size)


def refine_targets(queries, target_xy, target_grid, patch_size, mode="quadratic", temperature=0.05):
    """Sub-patch target localisation.

    quadratic: separable parabola through the similarities at (-1, 0, +1) patches.
    soft: softmax-weighted (temperature) mean offset over the 3x3 neighbourhood.
    Offsets are clipped to half a patch so a point never leaves its matched patch's
    Voronoi cell; invalid neighbours are ignored.
    """
    grid_features, grid_valid = target_grid
    g = grid_features.shape[0]
    out = target_xy.astype(np.float32).copy()
    for k, (q, xy) in enumerate(zip(queries, target_xy)):
        col = int(np.clip(np.floor(xy[0] / patch_size), 0, g - 1))
        row = int(np.clip(np.floor(xy[1] / patch_size), 0, g - 1))
        sims = np.full((3, 3), np.nan, dtype=np.float64)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if 0 <= r < g and 0 <= c < g and grid_valid[r, c]:
                    sims[dr + 1, dc + 1] = float(grid_features[r, c] @ q)
        if mode == "quadratic":
            ox = parabola_peak(sims[1, 0], sims[1, 1], sims[1, 2])
            oy = parabola_peak(sims[0, 1], sims[1, 1], sims[2, 1])
        elif mode == "soft":
            finite = np.isfinite(sims)
            weights = np.zeros_like(sims)
            values = sims[finite] / max(temperature, 1e-6)
            weights[finite] = np.exp(values - values.max())
            weights /= weights.sum()
            ys, xs = np.mgrid[-1:2, -1:2]
            ox, oy = float((weights * xs).sum()), float((weights * ys).sum())
        else:
            raise ValueError(f"Unknown refinement {mode}")
        out[k, 0] = xy[0] + np.clip(ox, -.5, .5) * patch_size
        out[k, 1] = xy[1] + np.clip(oy, -.5, .5) * patch_size
    return out


def parabola_peak(left, centre, right):
    """Offset (in samples) of the vertex of a parabola through 3 equally spaced samples."""
    if not (np.isfinite(left) and np.isfinite(centre) and np.isfinite(right)):
        return 0.0
    denom = left - 2 * centre + right
    if denom >= -1e-9:  # not a maximum
        return 0.0
    return float(0.5 * (left - right) / denom)


def sift_detect(image, valid, nfeatures=1500):
    detector = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), valid * 255)
    xy = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2)
    return xy, descriptors


def sift_matches(source, target, source_valid, target_valid, limit=400, ratio=0.8):
    left, descriptors_left = sift_detect(source, source_valid)
    right, descriptors_right = sift_detect(target, target_valid)
    empty = np.empty((0, 2), dtype=np.float32)
    if descriptors_left is None or descriptors_right is None or len(descriptors_right) < 2:
        return empty, empty.copy(), np.empty(0, np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates = matcher.knnMatch(descriptors_left, descriptors_right, k=2)
    accepted = [m for group in candidates if len(group) == 2 for m, n in [group] if m.distance < ratio * n.distance]
    if not accepted:
        return empty, empty.copy(), np.empty(0, np.float32)
    accepted.sort(key=lambda m: m.distance)
    accepted = accepted[:limit]
    return (np.array([left[m.queryIdx] for m in accepted], dtype=np.float32),
            np.array([right[m.trainIdx] for m in accepted], dtype=np.float32),
            np.array([1 / (1 + m.distance) for m in accepted], dtype=np.float32))


def dino_guided_sift(source, target, source_valid, target_valid, coarse_h, radius=24.0,
                     ratio=0.9, limit=400):
    """Coarse-to-fine: SIFT candidates restricted to a window around the DINOv3 homography.

    Falls back to plain SIFT when the coarse DINOv3 geometry failed.
    """
    if coarse_h is None:
        return sift_matches(source, target, source_valid, target_valid, limit)
    left, d_left = sift_detect(source, source_valid)
    right, d_right = sift_detect(target, target_valid)
    empty = np.empty((0, 2), dtype=np.float32)
    if d_left is None or d_right is None or not len(left) or not len(right):
        return empty, empty.copy(), np.empty(0, np.float32)
    predicted = project(coarse_h, left)
    src, dst, score = [], [], []
    for i, guess in enumerate(predicted):
        if not np.isfinite(guess).all():
            continue
        near = np.flatnonzero(np.linalg.norm(right - guess, axis=1) <= radius)
        if not len(near):
            continue
        distances = np.linalg.norm(d_right[near] - d_left[i], axis=1)
        order = np.argsort(distances)
        best = distances[order[0]]
        if len(order) > 1 and best >= ratio * distances[order[1]]:
            continue
        src.append(left[i])
        dst.append(right[near[order[0]]])
        score.append(1 / (1 + best))
    if not src:
        return empty, empty.copy(), np.empty(0, np.float32)
    order = np.argsort(-np.asarray(score))[:limit]
    return (np.asarray(src, np.float32)[order], np.asarray(dst, np.float32)[order],
            np.asarray(score, np.float32)[order])


def sift_dino_filter(source, target, source_valid, target_valid, source_grid, target_grid,
                     patch_size, top_k=5, ratio=0.9, limit=400):
    """SIFT with a relaxed ratio test, keeping only matches whose target patch is among the
    ``top_k`` most DINOv3-similar target patches of the source patch (semantic check)."""
    a, b, scores = sift_matches(source, target, source_valid, target_valid, limit=4 * limit, ratio=ratio)
    if not len(a):
        return a, b, scores
    fs, vs = source_grid
    ft, vt = target_grid
    g = fs.shape[0]
    flat_t = ft.reshape(g * g, -1)
    valid_t = vt.reshape(-1)
    keep = []
    for i, (p, q) in enumerate(zip(a, b)):
        rs, cs = (int(np.clip(p[1] // patch_size, 0, g - 1)), int(np.clip(p[0] // patch_size, 0, g - 1)))
        rt, ct = (int(np.clip(q[1] // patch_size, 0, g - 1)), int(np.clip(q[0] // patch_size, 0, g - 1)))
        sims = flat_t @ fs[rs, cs]
        sims[~valid_t] = -np.inf
        threshold = np.partition(sims, -top_k)[-top_k]
        if sims[rt * g + ct] >= threshold:
            keep.append(i)
    keep = np.asarray(keep[:limit], dtype=int)
    return a[keep], b[keep], scores[keep]


def estimate_geometry(source, target, threshold=3.0):
    if len(source) < 4:
        return None, 0
    matrix, inlier_mask = cv2.findHomography(np.asarray(source, np.float32),
                                             np.asarray(target, np.float32), cv2.RANSAC, threshold)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        return None, 0
    return matrix, int(inlier_mask.sum())


class DinoEncoder:
    def __init__(self, name, size, device):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.size = size
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.patch_size = int(self.model.config.patch_size)
        self.registers = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        if size % self.patch_size:
            raise ValueError(f"--size must be a multiple of model patch size {self.patch_size}")
        self.last_grid = None

    def encode(self, image, valid):
        """Return (valid patch features, their canvas centres); full grid kept in ``last_grid``."""
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = self.processor(images=Image.fromarray(rgb), return_tensors="pt",
                               do_resize=False, do_center_crop=False)["pixel_values"]
        if tuple(batch.shape[-2:]) != (self.size, self.size):
            raise RuntimeError("Processor resized the input unexpectedly")
        with self.torch.inference_mode():
            hidden = self.model(pixel_values=batch.to(self.device)).last_hidden_state[0]
        grid = self.size // self.patch_size
        tokens = hidden[1 + self.registers:]
        if len(tokens) != grid * grid:
            raise RuntimeError("Unexpected number of DINOv3 patch tokens")
        features = tokens.float().cpu().numpy()
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        ys, xs = np.indices((grid, grid))
        centers = np.column_stack(((xs.ravel() + .5) * self.patch_size,
                                   (ys.ravel() + .5) * self.patch_size)).astype(np.float32)
        idx = np.rint(centers).astype(int)
        keep = valid[np.clip(idx[:, 1], 0, self.size - 1),
                     np.clip(idx[:, 0], 0, self.size - 1)] > 0
        self.last_grid = (features.reshape(grid, grid, -1), keep.reshape(grid, grid))
        return features[keep], centers[keep]


class KorniaMatcher:
    """Learned-matcher baselines (optional dependency: ``pip install kornia``).

    Weights are fetched by kornia on first use. Untested in CI (needs torch + network).
    """

    def __init__(self, kind, device, limit=400):
        import torch
        import kornia.feature as KF

        self.torch, self.KF, self.kind, self.device, self.limit = torch, KF, kind, device, limit
        if kind == "LoFTR":
            self.model = KF.LoFTR(pretrained="outdoor").to(device).eval()
        elif kind == "LightGlue":
            self.disk = KF.DISK.from_pretrained("depth").to(device).eval()
            self.model = KF.LightGlueMatcher("disk").to(device).eval()
        else:
            raise ValueError(kind)

    def _tensor(self, image, gray):
        t = self.torch
        if gray:
            array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[None, None]
        else:
            array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        return t.from_numpy(np.ascontiguousarray(array)).float().div(255).to(self.device)

    def match(self, source, target, source_valid, target_valid):
        t = self.torch
        empty = np.empty((0, 2), np.float32)
        with t.inference_mode():
            if self.kind == "LoFTR":
                out = self.model({"image0": self._tensor(source, True), "image1": self._tensor(target, True)})
                a = out["keypoints0"].cpu().numpy()
                b = out["keypoints1"].cpu().numpy()
                s = out["confidence"].cpu().numpy()
            else:
                f0 = self.disk(self._tensor(source, False), 2048, pad_if_not_divisible=True)[0]
                f1 = self.disk(self._tensor(target, False), 2048, pad_if_not_divisible=True)[0]
                k0, k1 = f0.keypoints, f1.keypoints
                lafs0 = self.KF.laf_from_center_scale_ori(k0[None], t.ones(1, len(k0), 1, 1, device=self.device))
                lafs1 = self.KF.laf_from_center_scale_ori(k1[None], t.ones(1, len(k1), 1, 1, device=self.device))
                hw = t.tensor(source.shape[:2], device=self.device)
                dists, idxs = self.model(f0.descriptors, f1.descriptors, lafs0, lafs1, hw1=hw, hw2=hw)
                idxs = idxs.cpu().numpy()
                if not len(idxs):
                    return empty, empty.copy(), np.empty(0, np.float32)
                a = k0.cpu().numpy()[idxs[:, 0]]
                b = k1.cpu().numpy()[idxs[:, 1]]
                s = 1 - dists.cpu().numpy().reshape(-1)
        size = source_valid.shape[0]
        ia, ib = np.clip(np.rint(a).astype(int), 0, size - 1), np.clip(np.rint(b).astype(int), 0, size - 1)
        ok = (source_valid[ia[:, 1], ia[:, 0]] > 0) & (target_valid[ib[:, 1], ib[:, 0]] > 0)
        a, b, s = a[ok], b[ok], s[ok]
        order = np.argsort(-s)[:self.limit]
        return a[order].astype(np.float32), b[order].astype(np.float32), s[order].astype(np.float32)


def draw_example(path, source, target, source_valid, target_valid,
                 source_xy, target_xy, matrix, truth, inliers, method, error=None):
    size = source.shape[0]
    display = np.concatenate((source.copy(), target.copy()), axis=1)
    true_xy = project(truth, source_xy[:30])
    for a, b, expected in zip(source_xy[:30], target_xy[:30], true_xy):
        p0 = tuple(np.rint(a).astype(int))
        p1 = tuple(np.rint(b + [size, 0]).astype(int))
        ex, ey = np.rint(np.nan_to_num(expected, nan=-1e6,
                                      posinf=-1e6, neginf=-1e6)).astype(int)
        correct = (0 <= ex < size and 0 <= ey < size and target_valid[ey, ex] and
                   np.linalg.norm(b - expected) <= 5)
        cv2.line(display, p0, p1, (80, 210, 80) if correct else (40, 70, 230), 1)
    ys, xs = np.where(source_valid)
    corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()],
                        [xs.max(), ys.max()], [xs.min(), ys.max()]], np.float64)
    for homography, color in ((truth, (255, 0, 0)), (matrix, (0, 255, 0))):
        if homography is None:
            continue
        points = project(homography, corners)
        if np.isfinite(points).all():
            points[:, 0] += size
            cv2.polylines(display, [np.rint(points).astype(np.int32)], True, color, 2)
    title = f"{method}: {len(source_xy)} matches / {inliers} inliers"
    if error is not None:
        title += f" / {error:.2f}px"
    cv2.putText(display, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    cv2.putText(display, "Blue: HPatches GT   Green: estimated", (10, size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
    cv2.putText(display, "Match lines: green <=5px, red >5px (first 30)", (10, size - 34),
                cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1)
    cv2.imwrite(str(path), display)


def aggregate(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault((row["category"], row["method"]), [])
        group.append(row)
    summary = {}
    for (category, method), group in sorted(groups.items()):
        errors = [row["median_overlap_error_px"] for row in group if row["median_overlap_error_px"] is not None]
        n = len(group)
        summary[f"{category}/{method}"] = {
            "pairs": n, "registered": len(errors), "median_error_registered_px":
                float(np.median(errors)) if errors else None,
            "success_at_5px": sum(e is not None and e <= 5 for e in
                                  (row["median_overlap_error_px"] for row in group)) / n,
            "success_at_10px": sum(e is not None and e <= 10 for e in
                                   (row["median_overlap_error_px"] for row in group)) / n,
        }
    return summary


def aggregate_original(rows):
    """Literature-style numbers at original resolution: homography accuracy (corner error)
    and micro mean matching accuracy for 1..10px."""
    groups = {}
    for row in rows:
        groups.setdefault((row["category"], row["method"]), []).append(row)
    output = {}
    for (category, method), group in sorted(groups.items()):
        if "corner_error_orig_px" not in group[0]:
            continue
        errors = [row["corner_error_orig_px"] for row in group]
        finite = [e for e in errors if e is not None]
        total = sum(row["matches"] for row in group)
        item = {"pairs": len(group),
                "median_corner_error_orig_px": float(np.median(finite)) if finite else None}
        for t in HACC_THRESHOLDS:
            item[f"homography_accuracy_{t}px"] = sum(e is not None and e <= t for e in errors) / len(group)
        item["mma_micro"] = {str(t): (sum(row[f"correct_orig_{t}px"] for row in group) / total if total else None)
                             for t in MMA_THRESHOLDS}
        output[f"{category}/{method}"] = item
    return output


def by_target(rows):
    """Report target indices separately without assuming a monotonic difficulty."""
    output = {}
    for target in sorted({row["target"] for row in rows}):
        output[str(target)] = aggregate([row for row in rows if row["target"] == target])
    return output


def paired_comparison(rows, first="DINOv3", second="SIFT"):
    """Count wins only when both methods evaluated the same image pair."""
    indexed = {}
    for row in rows:
        indexed.setdefault((row["category"], row["sequence"], row["target"]), {})[row["method"]] = row["median_overlap_error_px"]
    counts = {f"{first}_better": 0, f"{second}_better": 0, "tie": 0, "both_failed": 0,
              "compared_pairs": 0}
    for methods in indexed.values():
        if first not in methods or second not in methods:
            continue
        counts["compared_pairs"] += 1
        a, b = methods[first], methods[second]
        if a is None and b is None:
            counts["both_failed"] += 1
        elif b is None or (a is not None and a < b - 1e-6):
            counts[f"{first}_better"] += 1
        elif a is None or b < a - 1e-6:
            counts[f"{second}_better"] += 1
        else:
            counts["tie"] += 1
    return counts


def aggregate_match_quality(rows):
    """Report both pooled match precision and equally weighted pair precision."""
    groups = {}
    for row in rows:
        groups.setdefault((row["category"], row["method"]), []).append(row)
    output = {}
    for (category, method), group in sorted(groups.items()):
        total = sum(row["matches"] for row in group)
        item = {"pairs": len(group), "pairs_with_matches": sum(row["matches"] > 0 for row in group),
                "proposed_matches": total,
                "gt_visible_source_matches": sum(row["gt_visible_matches"] for row in group)}
        for threshold in (5, 10, 16):
            correct = sum(row[f"correct_matches_{threshold}px"] for row in group)
            values = [row[f"match_precision_{threshold}px"] for row in group
                      if row[f"match_precision_{threshold}px"] is not None]
            item[f"correct_matches_{threshold}px"] = correct
            item[f"micro_precision_{threshold}px"] = correct / total if total else None
            item[f"mean_pair_precision_{threshold}px"] = float(np.mean(values)) if values else None
        output[f"{category}/{method}"] = item
    return output


def parse_visualize_pairs(values):
    wanted = set()
    for value in values or ():
        name, _, number = value.partition(":")
        if not name or not number.isdigit():
            raise ValueError(f"--visualize-pairs expects SEQUENCE:TARGET, got {value}")
        wanted.add((name, int(number)))
    return wanted


def run(args):
    opt = lambda name, default=None: getattr(args, name, default)  # noqa: E731  (v1 namespaces)
    methods = list(args.methods)
    pairs = discover_pairs(args.data, tuple(args.categories), args.max_sequences)
    print(f"Validated {len(pairs)} HPatches full-image pairs in {args.data}")
    if args.dry_run:
        return
    cv2.setRNGSeed(args.seed)
    np.random.seed(args.seed)
    ransac = args.ransac_threshold
    dino_ransac = opt("dino_ransac_threshold") or ransac
    refine = opt("refine", "none") or "none"
    perturb = opt("target_perturb", "none") or "none"
    perturb_px = opt("perturb_px", 7.0)
    guide_radius = opt("guide_radius", 24.0)
    want_visual = parse_visualize_pairs(opt("visualize_pairs"))
    encoder = None
    device = None
    if any(m in DINO_FEATURE_METHODS for m in methods):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {args.model} on {device}")
        encoder = DinoEncoder(args.model, args.size, device)
    learned = {}
    for kind in ("LoFTR", "LightGlue"):
        if kind in methods:
            import torch

            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            learned[kind] = KorniaMatcher(kind, device)
    patch_size = encoder.patch_size if encoder else 16
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    cached_ref = {}
    for index, (category, sequence, number, source_path, target_path, gt_path) in enumerate(pairs):
        source_original, target_original = cv2.imread(str(source_path)), cv2.imread(str(target_path))
        if source_original is None or target_original is None:
            raise ValueError(f"Cannot decode HPatches images in {sequence}")
        perturbation = make_perturbation(target_original.shape, perturb, perturb_px,
                                         stable_rng(args.seed, sequence, number))
        target_mask_original = None
        if perturbation is not None:
            target_original, target_mask_original = apply_perturbation(target_original, perturbation)
        source, transform0, valid0 = prepare_image(source_original, args.size)
        target, transform1, valid1 = prepare_image(target_original, args.size, target_mask_original)
        # Do not load the answer before every method has produced predictions.
        candidates = {}
        grids = {}
        if encoder is not None:
            if sequence not in cached_ref:
                feats = encoder.encode(source, valid0)
                cached_ref[sequence] = (feats, getattr(encoder, "last_grid", None))
            (features0, xy0), grids["source"] = cached_ref[sequence]
            features1, xy1 = encoder.encode(target, valid1)
            grids["target"] = getattr(encoder, "last_grid", None)
            dino = mutual_patch_matches(features0, xy0, features1, xy1, args.size, refine=refine,
                                        target_grid=grids["target"], patch_size=patch_size,
                                        temperature=opt("refine_temperature", 0.05))
            if "DINOv3" in methods:
                candidates["DINOv3"] = dino
            if "DINOguidedSIFT" in methods:
                coarse, _ = estimate_geometry(dino[0], dino[1], dino_ransac)
                candidates["DINOguidedSIFT"] = dino_guided_sift(source, target, valid0, valid1, coarse,
                                                                radius=guide_radius)
            if "SIFTdinoFilter" in methods:
                candidates["SIFTdinoFilter"] = sift_dino_filter(source, target, valid0, valid1, grids["source"],
                                                                grids["target"], patch_size)
        if "SIFT" in methods:
            candidates["SIFT"] = sift_matches(source, target, valid0, valid1)
        for kind, matcher in learned.items():
            candidates[kind] = matcher.match(source, target, valid0, valid1)
        estimates = {method: estimate_geometry(a, b, dino_ransac if method == "DINOv3" else ransac)
                     for method, (a, b, _) in candidates.items()}
        truth = load_ground_truth(gt_path, transform0, transform1, perturbation)
        truth_original = load_homography(gt_path)
        if perturbation is not None:
            truth_original = perturbation @ truth_original
        target_valid_original = (target_mask_original if target_mask_original is not None
                                 else np.ones(target_original.shape[:2], np.uint8))
        pair_rows = []
        for method in methods:
            a, b, scores = candidates[method]
            predicted, inliers = estimates[method]
            error = geometry_error(predicted, truth, valid0, valid1)
            quality = match_quality(a, b, truth, valid1)
            original_h = to_original_homography(predicted, transform0, transform1)
            row = {"category": category, "sequence": sequence, "target": number,
                   "method": method, "matches": len(scores), "inliers": inliers,
                   "median_overlap_error_px": error, "registered": error is not None,
                   **quality,
                   "corner_error_orig_px": corner_error(original_h, truth_original, source_original.shape),
                   **original_match_quality(a, b, transform0, transform1, truth_original,
                                            target_valid_original)}
            rows.append(row)
            pair_rows.append(row)
            if index < args.visualize or (sequence, number) in want_visual:
                draw_example(args.output / f"{sequence}_1_{number}_{method}.jpg",
                             source, target, valid0, valid1, a, b, predicted, truth, inliers, method, error)
        print(f"{index + 1}/{len(pairs)} {sequence} 1:{number}: " + ", ".join(
            f"{row['method']}={row['median_overlap_error_px']}" for row in pair_rows))
    with (args.output / "pairs.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = {}
    for other in methods:
        if other != "SIFT" and "SIFT" in methods:
            comparisons[f"{other}_vs_SIFT"] = paired_comparison(rows, other, "SIFT")
    report = {"dataset": str(args.data.resolve()), "protocol": "HPatches full images, reference 1 to targets 2..6",
              "categories": args.categories, "max_sequences_per_category": args.max_sequences,
              "model": args.model if encoder else None, "device": device,
              "methods": methods, "size": args.size,
              "patch_size": encoder.patch_size if encoder else None,
              "ransac_threshold_model_px": ransac, "dino_ransac_threshold_model_px": dino_ransac,
              "refine": refine, "target_perturb": perturb,
              "perturb_px_original": perturb_px if perturb != "none" else None,
              "guide_radius_model_px": guide_radius if "DINOguidedSIFT" in methods else None,
              "seed": args.seed,
              "metric": "median pixel difference between estimated and GT projections over fixed 17x17 grid in true visible overlap, resized canvas",
              "failure_policy": "failed registration counted as failure for success@5/10px; excluded from registered-only median",
              "match_metric": "all proposed matches denominator; GT-projected source outside visible target is incorrect; 5/10/16px in resized canvas; no recall claim",
              "original_metric": "corner error = mean distance of 4 source corners in ORIGINAL target pixels; MMA = correct/all proposed at 1..10 original px",
              "scope": "planar two-view registration, not metric 3D or video tracking",
              "summary": aggregate(rows), "by_target": by_target(rows),
              "paired_comparison": comparisons.get("DINOv3_vs_SIFT", paired_comparison(rows)),
              "paired_comparisons": comparisons,
              "match_quality": aggregate_match_quality(rows),
              "original_resolution": aggregate_original(rows)}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Extracted hpatches-sequences-release folder")
    parser.add_argument("--output", type=Path, default=Path("results/hpatches_spatial"))
    parser.add_argument("--categories", nargs="+", choices=("v", "i"), default=["v"])
    parser.add_argument("--max-sequences", type=int, help="First N sorted sequences per category (pilot only)")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS,
                        default=["DINOv3", "SIFT"], help="SIFT alone can smoke-test the pipeline without model weights")
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--dino-ransac-threshold", type=float,
                        help="RANSAC threshold for DINOv3 patch matches (default: --ransac-threshold)")
    parser.add_argument("--refine", choices=("none", "quadratic", "soft"), default="none",
                        help="sub-patch localisation of DINOv3 target points")
    parser.add_argument("--refine-temperature", type=float, default=0.05)
    parser.add_argument("--target-perturb", choices=("none", "shift", "homography"), default="none",
                        help="synthetic warp of target image; use with --categories i")
    parser.add_argument("--perturb-px", type=float, default=7.0,
                        help="shift: max |dx|,|dy|; homography: max corner displacement (original px)")
    parser.add_argument("--guide-radius", type=float, default=24.0,
                        help="DINOguidedSIFT search radius around the coarse DINOv3 homography (canvas px)")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--visualize", type=int, default=6)
    parser.add_argument("--visualize-pairs", nargs="*", default=[],
                        help="extra pairs to draw, e.g. v_bark:3 v_apprentices:6")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.size < 224 or args.ransac_threshold <= 0 or (args.max_sequences is not None and args.max_sequences < 1) \
            or args.visualize < 0 or (args.dino_ransac_threshold is not None and args.dino_ransac_threshold <= 0):
        parser.error("size must be >=224; thresholds/counts must be positive")
    run(args)  # DinoEncoder checks that --size is a multiple of the loaded model's patch size


if __name__ == "__main__":
    main()
