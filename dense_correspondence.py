"""Shared dense point-correspondence utilities (DINOv3 patch grid, dense SIFT, priors).

Used by spair_eval.py (semantic correspondence across instances) and
tapvid_davis_eval.py (same physical point across video frames). A query point is
described by bilinearly sampling the source patch-feature lattice; its match is the
most similar valid target patch, optionally refined inside the patch with the same
3x3 rule as hpatches_spatial_eval.refine_targets.
"""

import cv2
import numpy as np

from hpatches_spatial_eval import refine_targets


class Geometry:
    """Original image <-> square model canvas. mode='letterbox' keeps aspect, 'stretch' does not."""

    def __init__(self, shape, size, mode="letterbox"):
        height, width = shape[:2]
        self.size, self.mode = size, mode
        if mode == "stretch":
            self.sx, self.sy, self.left, self.top = size / width, size / height, 0.0, 0.0
            self.w, self.h = size, size
        else:
            scale = min(size / width, size / height)
            self.w, self.h = max(1, round(width * scale)), max(1, round(height * scale))
            self.sx, self.sy = self.w / width, self.h / height
            self.left, self.top = (size - self.w) // 2, (size - self.h) // 2

    def to_canvas(self, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        return np.column_stack((xy[:, 0] * self.sx + self.left, xy[:, 1] * self.sy + self.top))

    def to_original(self, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        return np.column_stack(((xy[:, 0] - self.left) / self.sx, (xy[:, 1] - self.top) / self.sy))

    def canvas(self, bgr):
        resized = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        out = np.zeros((self.size, self.size, 3), np.uint8)
        out[int(self.top):int(self.top) + self.h, int(self.left):int(self.left) + self.w] = resized
        valid = np.zeros((self.size, self.size), np.uint8)
        valid[int(self.top):int(self.top) + self.h, int(self.left):int(self.left) + self.w] = 1
        return out, valid


class DenseDino:
    def __init__(self, name, size, device=None, mode="letterbox"):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.size, self.mode = size, mode
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.patch_size = int(self.model.config.patch_size)
        self.registers = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        if size % self.patch_size:
            raise ValueError(f"size must be a multiple of patch size {self.patch_size}")
        self.grid = size // self.patch_size

    def encode(self, bgr):
        """Return (features[g,g,C] L2-normalised, valid[g,g] bool, Geometry)."""
        from PIL import Image

        geo = Geometry(bgr.shape, self.size, self.mode)
        canvas, valid = geo.canvas(bgr)
        pixels = self.processor(images=Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)),
                                return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"]
        with self.torch.inference_mode():
            hidden = self.model(pixel_values=pixels.to(self.device)).last_hidden_state[0]
        tokens = hidden[1 + self.registers:].float().cpu().numpy()
        if len(tokens) != self.grid ** 2:
            raise RuntimeError("Unexpected patch token count")
        tokens /= np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-12)
        return tokens.reshape(self.grid, self.grid, -1), patch_valid(valid, self.patch_size), geo


def patch_valid(valid, patch_size):
    g = valid.shape[0] // patch_size
    centres = (np.arange(g) + .5) * patch_size
    idx = np.clip(np.rint(centres).astype(int), 0, valid.shape[0] - 1)
    return valid[np.ix_(idx, idx)] > 0


def sample_features(grid, xy_canvas, patch_size):
    """Bilinear sampling on the patch-centre lattice; returns L2-normalised (N, C)."""
    g = grid.shape[0]
    u = np.clip(np.asarray(xy_canvas)[:, 0] / patch_size - .5, 0, g - 1)
    v = np.clip(np.asarray(xy_canvas)[:, 1] / patch_size - .5, 0, g - 1)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    u1, v1 = np.minimum(u0 + 1, g - 1), np.minimum(v0 + 1, g - 1)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    out = (grid[v0, u0] * (1 - fu) * (1 - fv) + grid[v0, u1] * fu * (1 - fv) +
           grid[v1, u0] * (1 - fu) * fv + grid[v1, u1] * fu * fv)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def nearest_patch(queries, grid, valid, patch_size, refine="none", temperature=0.05, region=None):
    """Most similar valid target patch for each query (canvas coordinates of the match).

    ``region`` (optional bool[g,g]) further restricts candidates, e.g. a target bounding box.
    """
    g = grid.shape[0]
    flat = grid.reshape(g * g, -1)
    allowed = valid.reshape(-1) if region is None else (valid & region).reshape(-1)
    if not allowed.any():
        allowed = valid.reshape(-1)
    sims = queries @ flat.T
    sims[:, ~allowed] = -np.inf
    best = sims.argmax(axis=1)
    rows, cols = best // g, best % g
    xy = np.column_stack(((cols + .5) * patch_size, (rows + .5) * patch_size)).astype(np.float32)
    if refine and refine != "none":
        xy = refine_targets(queries, xy, (grid, valid), patch_size, refine, temperature)
    return xy, sims[np.arange(len(best)), best]


class DenseSift:
    """Hand-crafted baseline: SIFT descriptor at the query vs a dense SIFT grid on the target."""

    def __init__(self, step=4, scale=16.0):
        self.sift = cv2.SIFT_create()
        self.step, self.scale = step, scale

    def describe(self, gray, xy):
        keypoints = [cv2.KeyPoint(float(x), float(y), self.scale, -1, 0, 0, i) for i, (x, y) in enumerate(xy)]
        kept, desc = self.sift.compute(gray, keypoints)
        out = np.zeros((len(xy), 128), np.float32)
        if desc is not None:  # OpenCV may drop points; re-align by class_id
            for kp, row in zip(kept, desc):
                out[kp.class_id] = row
        return out

    def match(self, gray0, xy0, gray1, region_mask=None):
        h, w = gray1.shape
        ys, xs = np.mgrid[self.step // 2:h:self.step, self.step // 2:w:self.step]
        grid_xy = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
        if region_mask is not None:
            keep = region_mask[grid_xy[:, 1].astype(int), grid_xy[:, 0].astype(int)] > 0
            if keep.any():
                grid_xy = grid_xy[keep]
        d0 = self.describe(gray0, xy0)
        d1 = self.describe(gray1, grid_xy)
        d2 = (d0 ** 2).sum(1)[:, None] + (d1 ** 2).sum(1)[None] - 2 * d0 @ d1.T
        return grid_xy[d2.argmin(axis=1)]
