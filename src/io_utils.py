from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def imread(path: str | Path, unchanged: bool = True) -> np.ndarray:
    """Unicode-safe image reader returning BGR/BGRA arrays."""
    path = Path(path)
    flags = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        try:
            pil = np.asarray(Image.open(path))
            if pil.ndim == 2:
                image = pil.copy()
            elif pil.shape[2] == 4:
                image = cv2.cvtColor(pil, cv2.COLOR_RGBA2BGRA)
            else:
                image = cv2.cvtColor(pil, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"Cannot read image: {path}") from exc
    return image


def imwrite(path: str | Path, image: np.ndarray) -> None:
    """Unicode-safe image writer, preserving TGA when requested."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".tga":
        if image.ndim == 2:
            pil = Image.fromarray(image)
        elif image.shape[2] == 4:
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        else:
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil.save(path)
        return
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def as_bgr(image: np.ndarray, background: int = 255) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    return np.rint(image[:, :, :3] * alpha + background * (1.0 - alpha)).astype(np.uint8)
