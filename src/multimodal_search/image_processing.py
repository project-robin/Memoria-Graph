from __future__ import annotations

import base64
import hashlib
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError

from multimodal_search.config import FACE_DIR, SUPPORTED_MIME_TYPES, TEMP_IMAGE_DIR, THUMBNAIL_DIR, THUMBNAIL_MAX_EDGE

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency during tests
    cv2 = None


class ImageProcessingError(RuntimeError):
    """Raised when a source image cannot be prepared for indexing."""


def resolve_mime_type(path: str, uploaded_mime_type: str | None = None) -> str:
    if uploaded_mime_type in SUPPORTED_MIME_TYPES:
        return uploaded_mime_type
    guessed, _ = mimetypes.guess_type(path)
    if guessed in SUPPORTED_MIME_TYPES:
        return guessed
    raise ImageProcessingError(f"Unsupported image type for '{path}'.")


def validate_image_bytes(image_bytes: bytes) -> None:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except UnidentifiedImageError as exc:
        raise ImageProcessingError("Uploaded file is not a valid image.") from exc
    except Exception as exc:
        raise ImageProcessingError(f"Image validation failed: {exc}") from exc


def save_uploaded_source(filename: str, image_bytes: bytes, mime_type: str) -> Path:
    suffix = Path(filename).suffix.lower() or mimetypes.guess_extension(mime_type) or ".img"
    destination = TEMP_IMAGE_DIR / f"{uuid4().hex}{suffix}"
    destination.write_bytes(image_bytes)
    return destination


def discover_image_files(folder_path: str) -> list[str]:
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise ImageProcessingError(f"Folder not found: {folder}")

    paths: list[str] = []
    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            resolve_mime_type(str(file_path))
        except ImageProcessingError:
            continue
        paths.append(str(file_path))
    return sorted(paths)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _extract_exif(image: Image.Image) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        exif = image.getexif()
    except Exception:
        exif = None

    if not exif:
        return metadata

    date_time = exif.get(306) or exif.get(36867) or exif.get(36868)
    if isinstance(date_time, str):
        normalized = date_time.replace(":", "-", 2).replace(" ", "T", 1)
        metadata["exif_datetime"] = normalized
        if len(normalized) >= 7:
            metadata["exif_year"] = int(normalized[:4])
            metadata["exif_month"] = int(normalized[5:7])

    gps_info = exif.get_ifd(34853) if hasattr(exif, "get_ifd") else None
    if gps_info:
        lat = _gps_to_decimal(gps_info.get(2), gps_info.get(1))
        lng = _gps_to_decimal(gps_info.get(4), gps_info.get(3))
        if lat is not None:
            metadata["gps_lat"] = lat
        if lng is not None:
            metadata["gps_lng"] = lng
    return metadata


def _gps_to_decimal(values: Any, ref: Any) -> float | None:
    if not values:
        return None
    try:
        deg = float(values[0][0]) / float(values[0][1])
        minutes = float(values[1][0]) / float(values[1][1])
        seconds = float(values[2][0]) / float(values[2][1])
        decimal = deg + (minutes / 60) + (seconds / 3600)
        if ref in {"S", "W"}:
            decimal *= -1
        return decimal
    except Exception:
        return None


def prepare_image_asset(source_path: str) -> dict[str, Any]:
    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        raise ImageProcessingError(f"Missing file: {path}")

    mime_type = resolve_mime_type(str(path))
    image_bytes = path.read_bytes()
    validate_image_bytes(image_bytes)

    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        width, height = image.size
        exif = _extract_exif(image)

        thumbnail = image.copy()
        thumbnail.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE))
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format="JPEG", quality=88)
        thumb_bytes = thumb_buffer.getvalue()

    source_hash = _sha256_bytes(image_bytes)
    thumbnail_path = THUMBNAIL_DIR / f"{source_hash}.jpg"
    if not thumbnail_path.exists():
        thumbnail_path.write_bytes(thumb_bytes)

    return {
        "source_path": str(path),
        "filename": path.name,
        "album": path.parent.name,
        "mime_type": mime_type,
        "source_hash": source_hash,
        "width": width,
        "height": height,
        "thumbnail_path": str(thumbnail_path),
        "thumbnail_bytes": thumb_bytes,
        "thumbnail_b64": base64.b64encode(thumb_bytes).decode("ascii"),
        **exif,
    }


def decode_thumbnail_bytes(thumbnail_b64: str) -> bytes:
    return base64.b64decode(thumbnail_b64.encode("ascii"))


def detect_face_crops(thumbnail_path: str, image_id: str) -> list[dict[str, Any]]:
    if cv2 is None:
        return []

    image = cv2.imread(thumbnail_path)
    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    detected: list[dict[str, Any]] = []
    for index, (x, y, w, h) in enumerate(faces):
        crop = image[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb).resize((128, 128))
        crop_path = FACE_DIR / f"{image_id}_{index}.jpg"
        pil_crop.save(crop_path, format="JPEG", quality=90)
        detected.append(
            {
                "face_path": str(crop_path),
                "face_hash": dhash(pil_crop),
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            }
        )
    return detected


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    grayscale = image.convert("L").resize((hash_size + 1, hash_size))
    diff = np.array(grayscale)[:, 1:] > np.array(grayscale)[:, :-1]
    bit_string = "".join("1" if value else "0" for value in diff.flatten())
    return f"{int(bit_string, 2):0{hash_size * hash_size // 4}x}"
