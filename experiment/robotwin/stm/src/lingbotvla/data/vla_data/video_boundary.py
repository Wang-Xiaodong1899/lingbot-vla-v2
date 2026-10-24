"""Episode-local video queries. Missing tail images repeat the last valid frame."""
import math


def episode_video_queries(timestamps, start, end, fps):
    start, end, fps = float(start), float(end), float(fps)
    if not all(math.isfinite(x) for x in (start, end, fps)) or fps <= 0 or end <= start:
        raise ValueError(f"Invalid episode video interval: {(start, end, fps)}")
    frames = int(round((end - start) * fps))
    if frames < 1 or abs((end-start)*fps - frames) > 0.02:
        raise ValueError("Episode video interval is not aligned to the declared frame rate")
    # Metadata end is exclusive. Quantize relative timestamps in float64 before
    # adding the shared-file offset; never clamp against the whole file length.
    indices = [max(0, min(int(round(float(t)*fps)), frames-1)) for t in timestamps]
    return [start + i/fps for i in indices], (start, end)
