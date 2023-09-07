# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial

import torch
import numpy as np

from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
    """
    if hasattr(predictor, 'trackers') and persist:
        return
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))
    assert cfg.tracker_type in ['bytetrack', 'botsort'], \
        f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
    predictor.trackers = trackers


def on_predict_postprocess_end(predictor):
    """Postprocess detected boxes and update with object tracking."""
    bs = predictor.dataset.bs
    im0s = predictor.batch[1]
    for i in range(bs):
        det = predictor.results[i].boxes.cpu().numpy()
        if len(det) == 0:
            continue
        tracks = predictor.trackers[i].update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        ball_idx = predictor.results[i].boxes.cls == 0 # Captura os índices das detecções da bola
        if True in ball_idx: # Há detecções da bola
            # Armazena as bounding boxes das detecções da bola
            ball_detections = predictor.results[i][ball_idx]
            ball_bboxes = ball_detections.boxes.data.cpu().numpy()
            # Altera o array das bboxes para ser compatível com o array do tracking
            ball_bboxes = np.insert(ball_bboxes, 4, -1, axis=1) # xyxy, track_id = -1, conf, cls
            ball_bboxes = np.append(ball_bboxes, np.full((ball_bboxes.shape[0], 1), -1), axis=1) # xyxy, track_id = -1, conf, cls, idx = -1
            # Junta o Tracking dos Jogadores com as detecções da bola
            tracks = np.concatenate((tracks, ball_bboxes))
        predictor.results[i] = predictor.results[i][idx]
        predictor.results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))


def register_tracker(model, persist):
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    """
    model.add_callback('on_predict_start', partial(on_predict_start, persist=persist))
    model.add_callback('on_predict_postprocess_end', on_predict_postprocess_end)
