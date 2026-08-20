from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any

from .fusion import iou
from .types import Box, Detection


def identity_affine() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def cosine_similarity(first: list[float] | None, second: list[float] | None) -> float | None:
    if first is None or second is None or len(first) != len(second) or not first:
        return None
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return None
    return max(-1.0, min(1.0, dot / (first_norm * second_norm)))


def _normalized_embedding(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def _label_tokens(label: str) -> set[str]:
    return {token for token in label.lower().replace("/", " ").split() if token}


def transform_box(
    box: Box,
    affine: tuple[tuple[float, float, float], tuple[float, float, float]],
    width: int,
    height: int,
) -> Box:
    points = ((box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2))
    transformed = [
        (
            affine[0][0] * x + affine[0][1] * y + affine[0][2],
            affine[1][0] * x + affine[1][1] * y + affine[1][2],
        )
        for x, y in points
    ]
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    return Box(min(xs), min(ys), max(xs), max(ys)).clip(width, height)


class GlobalMotionEstimator:
    """Estimate camera translation/rotation/scale with sparse optical flow.

    The tracker remains functional without OpenCV, falling back to identity
    camera motion and each track's learned residual velocity.
    """

    def __init__(self, max_width: int = 960, prefer_opencv: bool = True) -> None:
        self.max_width = max_width
        cv2 = None
        if prefer_opencv:
            try:
                import cv2
            except ImportError:
                cv2 = None
        self.cv2 = cv2

    @property
    def available(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "OpenCV sparse optical-flow affine" if self.cv2 is not None else "NumPy phase-correlation translation"

    def _phase_correlation(self, previous_gray: Any, current_gray: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        import numpy as np
        from PIL import Image

        source_height, source_width = previous_gray.shape[:2]
        scale = min(1.0, self.max_width / max(1, source_width))
        if scale < 1.0:
            size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
            previous = np.asarray(Image.fromarray(previous_gray).resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
            current = np.asarray(Image.fromarray(current_gray).resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
        else:
            previous = previous_gray.astype(np.float32)
            current = current_gray.astype(np.float32)
        previous -= float(previous.mean())
        current -= float(current.mean())
        window = np.outer(np.hanning(previous.shape[0]), np.hanning(previous.shape[1])).astype(np.float32)
        previous *= window
        current *= window
        previous_frequency = np.fft.fft2(previous)
        current_frequency = np.fft.fft2(current)
        cross_power = current_frequency * np.conj(previous_frequency)
        magnitude = np.abs(cross_power)
        cross_power /= np.maximum(magnitude, 1e-9)
        correlation = np.abs(np.fft.ifft2(cross_power))
        peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
        peak_ratio = float(correlation[peak_y, peak_x] / max(1e-9, correlation.mean()))
        if peak_ratio < 5.0:
            return identity_affine()
        if peak_x > correlation.shape[1] // 2:
            peak_x -= correlation.shape[1]
        if peak_y > correlation.shape[0] // 2:
            peak_y -= correlation.shape[0]
        tx = float(peak_x) / scale
        ty = float(peak_y) / scale
        if abs(tx) > source_width * 0.25 or abs(ty) > source_height * 0.25:
            return identity_affine()
        return ((1.0, 0.0, tx), (0.0, 1.0, ty))

    def estimate(self, previous_gray: Any, current_gray: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if previous_gray is None or current_gray is None:
            return identity_affine()
        if self.cv2 is None:
            return self._phase_correlation(previous_gray, current_gray)
        cv2 = self.cv2
        source_height, source_width = previous_gray.shape[:2]
        scale = min(1.0, self.max_width / max(1, source_width))
        if scale < 1.0:
            size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
            previous = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
            current = cv2.resize(current_gray, size, interpolation=cv2.INTER_AREA)
        else:
            previous, current = previous_gray, current_gray

        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if points is None or len(points) < 8:
            return identity_affine()
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if moved is None or status is None:
            return identity_affine()
        valid = status.reshape(-1).astype(bool)
        if int(valid.sum()) < 8:
            return identity_affine()
        matrix, _ = cv2.estimateAffinePartial2D(
            points.reshape(-1, 2)[valid],
            moved.reshape(-1, 2)[valid],
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if matrix is None:
            return identity_affine()

        a, b, tx = (float(value) for value in matrix[0])
        c, d, ty = (float(value) for value in matrix[1])
        estimated_scale = math.sqrt(max(0.0, a * a + c * c))
        if not 0.90 <= estimated_scale <= 1.10:
            return identity_affine()
        if abs(tx) > previous.shape[1] * 0.25 or abs(ty) > previous.shape[0] * 0.25:
            return identity_affine()
        if scale < 1.0:
            tx /= scale
            ty /= scale
        return ((a, b, tx), (c, d, ty))


@dataclass
class TrackState:
    track_id: int
    label: str
    box: Box
    score_ema: float
    first_frame: int
    last_frame: int
    last_detection_frame: int
    hits: int = 1
    missed_keyframes: int = 0
    confirmed: bool = False
    residual_velocity: tuple[float, float] = (0.0, 0.0)
    source_models: set[str] = field(default_factory=set)
    scores: list[float] = field(default_factory=list)
    heights_px: list[float] = field(default_factory=list)
    detected_frames: list[int] = field(default_factory=list)
    appearance_embedding: list[float] | None = field(default=None, repr=False)
    appearance_encoder: str | None = None
    appearance_observations: int = 0
    appearance_reliability_ema: float = 0.0
    appearance_similarity_sum: float = 0.0
    appearance_similarity_count: int = 0
    last_appearance_similarity: float | None = None
    best_quality: float = -1.0
    best_frame: int | None = None
    best_crop: Any = field(default=None, repr=False)
    best_crop_path: str | None = None

    @classmethod
    def from_detection(cls, track_id: int, detection: Detection, frame_index: int, min_hits: int) -> "TrackState":
        sources = detection.metadata.get("source_models", [detection.model])
        embedding = detection.metadata.get("appearance_embedding")
        reliability = float(detection.metadata.get("appearance_reliability", 0.0))
        return cls(
            track_id=track_id,
            label=detection.label,
            box=detection.box,
            score_ema=detection.score,
            first_frame=frame_index,
            last_frame=frame_index,
            last_detection_frame=frame_index,
            confirmed=min_hits <= 1,
            source_models={str(value) for value in sources},
            scores=[detection.score],
            heights_px=[detection.box.height],
            detected_frames=[frame_index],
            appearance_embedding=list(embedding) if embedding is not None else None,
            appearance_encoder=detection.metadata.get("appearance_encoder"),
            appearance_observations=1 if embedding is not None else 0,
            appearance_reliability_ema=reliability,
        )

    def propagate(
        self,
        affine: tuple[tuple[float, float, float], tuple[float, float, float]],
        frame_index: int,
        width: int,
        height: int,
    ) -> None:
        camera_box = transform_box(self.box, affine, width, height)
        dx, dy = self.residual_velocity
        self.box = camera_box.translate(dx, dy).clip(width, height)
        self.last_frame = frame_index

    def update_detection(
        self,
        detection: Detection,
        frame_index: int,
        min_hits: int,
        appearance_momentum: float = 0.85,
    ) -> None:
        elapsed = max(1, frame_index - self.last_detection_frame)
        predicted_center = ((self.box.x1 + self.box.x2) / 2.0, (self.box.y1 + self.box.y2) / 2.0)
        detected_center = ((detection.box.x1 + detection.box.x2) / 2.0, (detection.box.y1 + detection.box.y2) / 2.0)
        residual = (
            (detected_center[0] - predicted_center[0]) / elapsed,
            (detected_center[1] - predicted_center[1]) / elapsed,
        )
        self.residual_velocity = (
            0.6 * self.residual_velocity[0] + 0.4 * residual[0],
            0.6 * self.residual_velocity[1] + 0.4 * residual[1],
        )
        self.box = detection.box
        self.score_ema = 0.7 * self.score_ema + 0.3 * detection.score
        self.last_frame = frame_index
        self.last_detection_frame = frame_index
        self.hits += 1
        self.missed_keyframes = 0
        self.confirmed = self.confirmed or self.hits >= min_hits
        sources = detection.metadata.get("source_models", [detection.model])
        self.source_models.update(str(value) for value in sources)
        self.scores.append(detection.score)
        self.heights_px.append(detection.box.height)
        self.detected_frames.append(frame_index)
        embedding = detection.metadata.get("appearance_embedding")
        if embedding is not None:
            candidate = [float(value) for value in embedding]
            similarity = cosine_similarity(self.appearance_embedding, candidate)
            self.last_appearance_similarity = similarity
            if similarity is not None:
                self.appearance_similarity_sum += similarity
                self.appearance_similarity_count += 1
            if self.appearance_embedding is None or len(self.appearance_embedding) != len(candidate):
                self.appearance_embedding = _normalized_embedding(candidate)
            else:
                mixed = [
                    appearance_momentum * old + (1.0 - appearance_momentum) * new
                    for old, new in zip(self.appearance_embedding, candidate, strict=True)
                ]
                self.appearance_embedding = _normalized_embedding(mixed)
            reliability = float(detection.metadata.get("appearance_reliability", 0.0))
            if self.appearance_observations == 0:
                self.appearance_reliability_ema = reliability
            else:
                self.appearance_reliability_ema = (
                    appearance_momentum * self.appearance_reliability_ema
                    + (1.0 - appearance_momentum) * reliability
                )
            self.appearance_encoder = str(
                detection.metadata.get("appearance_encoder", self.appearance_encoder or "unknown")
            )
            self.appearance_observations += 1

    def temporal_confidence(self, min_hits: int) -> float:
        evidence = min(1.0, self.hits / max(1, min_hits))
        model_support = min(1.0, 0.75 + 0.25 * len(self.source_models))
        return min(1.0, self.score_ema * evidence * model_support)

    def to_frame_dict(self, detected_this_frame: bool, min_hits: int) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "box_xyxy": self.box.to_list(),
            "object_height_px": self.box.height,
            "score_ema": self.score_ema,
            "temporal_confidence": self.temporal_confidence(min_hits),
            "hits": self.hits,
            "confirmed": self.confirmed,
            "detected_this_frame": detected_this_frame,
            "source_models": sorted(self.source_models),
            "appearance_observations": self.appearance_observations,
            "appearance_reliability": self.appearance_reliability_ema,
            "last_appearance_similarity": self.last_appearance_similarity,
        }

    def to_summary_dict(self, fps: float, min_hits: int) -> dict[str, Any]:
        mean_score = sum(self.scores) / len(self.scores) if self.scores else 0.0
        sorted_heights = sorted(self.heights_px)
        median_height = sorted_heights[len(sorted_heights) // 2] if sorted_heights else 0.0
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confirmed": self.confirmed,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_time_seconds": self.first_frame / fps,
            "last_time_seconds": self.last_frame / fps,
            "duration_seconds": max(0.0, (self.last_frame - self.first_frame + 1) / fps),
            "detection_hits": self.hits,
            "detected_frames": self.detected_frames,
            "mean_detection_score": mean_score,
            "maximum_detection_score": max(self.scores, default=0.0),
            "temporal_confidence": self.temporal_confidence(min_hits),
            "median_detected_height_px": median_height,
            "minimum_detected_height_px": min(self.heights_px, default=0.0),
            "maximum_detected_height_px": max(self.heights_px, default=0.0),
            "source_models": sorted(self.source_models),
            "appearance_encoder": self.appearance_encoder,
            "appearance_observations": self.appearance_observations,
            "mean_appearance_similarity": (
                self.appearance_similarity_sum / self.appearance_similarity_count
                if self.appearance_similarity_count
                else None
            ),
            "appearance_reliability": self.appearance_reliability_ema,
            "best_frame": self.best_frame,
            "best_quality_score": self.best_quality,
            "best_crop_path": self.best_crop_path,
        }


class TemporalTracker:
    """Persistent object-query baseline with geometry-aware temporal memory."""

    def __init__(
        self,
        *,
        min_hits: int = 2,
        max_missed_keyframes: int = 2,
        association_iou: float = 0.2,
        appearance_weight: float = 0.35,
        appearance_min_similarity: float = 0.25,
        appearance_cross_label_similarity: float = 0.90,
        appearance_momentum: float = 0.85,
    ) -> None:
        if min_hits < 1 or max_missed_keyframes < 0:
            raise ValueError("min_hits must be positive and max_missed_keyframes cannot be negative")
        self.min_hits = min_hits
        self.max_missed_keyframes = max_missed_keyframes
        self.association_iou = association_iou
        self.appearance_weight = appearance_weight
        self.appearance_min_similarity = appearance_min_similarity
        self.appearance_cross_label_similarity = appearance_cross_label_similarity
        self.appearance_momentum = appearance_momentum
        self.active: dict[int, TrackState] = {}
        self.finished: list[TrackState] = []
        self.next_track_id = 1

    def _association_score(self, track: TrackState, detection: Detection) -> float | None:
        exact_label = track.label.lower() == detection.label.lower()
        related_label = bool(_label_tokens(track.label) & _label_tokens(detection.label))
        overlap = iou(track.box, detection.box)
        track_center = ((track.box.x1 + track.box.x2) / 2.0, (track.box.y1 + track.box.y2) / 2.0)
        detection_center = ((detection.box.x1 + detection.box.x2) / 2.0, (detection.box.y1 + detection.box.y2) / 2.0)
        distance = math.dist(track_center, detection_center)
        normalizer = max(20.0, math.hypot(track.box.width, track.box.height), math.hypot(detection.box.width, detection.box.height))
        normalized_distance = distance / normalizer
        if overlap <= 0.0 and normalized_distance > 0.75:
            return None
        detection_embedding = detection.metadata.get("appearance_embedding")
        appearance = cosine_similarity(track.appearance_embedding, detection_embedding)
        detection_reliability = float(detection.metadata.get("appearance_reliability", 0.0))
        reliability = math.sqrt(max(0.0, track.appearance_reliability_ema * detection_reliability))

        if not (exact_label or related_label):
            if (
                appearance is None
                or appearance < self.appearance_cross_label_similarity
                or normalized_distance > 0.45
            ):
                return None
        elif (
            appearance is not None
            and appearance < self.appearance_min_similarity
            and overlap < max(0.5, self.association_iou)
        ):
            return None

        score = overlap + 0.20 * max(0.0, 1.0 - normalized_distance)
        if appearance is not None:
            score += self.appearance_weight * max(0.0, appearance) * reliability
        if not exact_label:
            score -= 0.05 if related_label else 0.15
        return score

    def update(
        self,
        *,
        frame_index: int,
        width: int,
        height: int,
        affine: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        detections: list[Detection] | None = None,
        detection_frame: bool = False,
    ) -> tuple[list[TrackState], set[int]]:
        transform = affine or identity_affine()
        for track in self.active.values():
            if frame_index > track.last_frame:
                track.propagate(transform, frame_index, width, height)

        detected_track_ids: set[int] = set()
        if detection_frame:
            current_detections = detections or []
            candidates: list[tuple[float, int, int]] = []
            for track_id, track in self.active.items():
                for detection_index, detection in enumerate(current_detections):
                    score = self._association_score(track, detection)
                    if score is not None and (iou(track.box, detection.box) >= self.association_iou or score >= 0.12):
                        candidates.append((score, track_id, detection_index))

            matched_tracks: set[int] = set()
            matched_detections: set[int] = set()
            for _, track_id, detection_index in sorted(candidates, reverse=True):
                if track_id in matched_tracks or detection_index in matched_detections:
                    continue
                self.active[track_id].update_detection(
                    current_detections[detection_index],
                    frame_index,
                    self.min_hits,
                    self.appearance_momentum,
                )
                matched_tracks.add(track_id)
                matched_detections.add(detection_index)
                detected_track_ids.add(track_id)

            for track_id, track in list(self.active.items()):
                if track_id not in matched_tracks:
                    track.missed_keyframes += 1
                    if track.missed_keyframes > self.max_missed_keyframes:
                        self.finished.append(track)
                        del self.active[track_id]

            for detection_index, detection in enumerate(current_detections):
                if detection_index in matched_detections:
                    continue
                track = TrackState.from_detection(self.next_track_id, detection, frame_index, self.min_hits)
                self.active[track.track_id] = track
                detected_track_ids.add(track.track_id)
                self.next_track_id += 1

        return sorted(self.active.values(), key=lambda item: item.track_id), detected_track_ids

    def all_tracks(self) -> list[TrackState]:
        return sorted([*self.finished, *self.active.values()], key=lambda item: item.track_id)

    def counts_by_label(self, confirmed_only: bool = True) -> dict[str, int]:
        return dict(
            Counter(
                track.label
                for track in self.all_tracks()
                if track.confirmed or not confirmed_only
            )
        )
