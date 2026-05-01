import json
import os
import time
from typing import List

import mss
import numpy as np
from PIL import Image

from openrecall.config import args, screenshots_path
from openrecall.database import insert_entry
from openrecall.nlp import get_embedding
from openrecall.ocr_normalize import normalize_ocr_text
from openrecall.ocr import extract_ocr_result_from_image
from openrecall.utils import (
    get_active_app_name,
    get_active_window_title,
    is_user_active,
)


def mean_structured_similarity_index(
    img1: np.ndarray, img2: np.ndarray, L: int = 255
) -> float:
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    def rgb2gray(img: np.ndarray) -> np.ndarray:
        return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

    img1_gray: np.ndarray = rgb2gray(img1)
    img2_gray: np.ndarray = rgb2gray(img2)
    mu1: float = np.mean(img1_gray)
    mu2: float = np.mean(img2_gray)
    sigma1_sq = np.var(img1_gray)
    sigma2_sq = np.var(img2_gray)
    sigma12 = np.mean((img1_gray - mu1) * (img2_gray - mu2))
    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )


def is_similar(
    img1: np.ndarray, img2: np.ndarray, similarity_threshold: float = 0.9
) -> bool:
    return mean_structured_similarity_index(img1, img2) >= similarity_threshold


def take_screenshots() -> List[np.ndarray]:
    screenshots: List[np.ndarray] = []
    with mss.mss() as sct:
        monitor_indices = range(1, len(sct.monitors))
        if args.primary_monitor_only:
            monitor_indices = [1]

        for index in monitor_indices:
            if index >= len(sct.monitors):
                print(f"Warning: Monitor index {index} out of bounds. Skipping.")
                continue
            screenshot = np.array(sct.grab(sct.monitors[index]))[:, :, [2, 1, 0]]
            screenshots.append(screenshot)

    return screenshots


def record_screenshots_thread() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    last_screenshots: List[np.ndarray] = take_screenshots()

    while True:
        if not is_user_active():
            time.sleep(3)
            continue

        current_screenshots: List[np.ndarray] = take_screenshots()
        if len(last_screenshots) != len(current_screenshots):
            last_screenshots = current_screenshots
            time.sleep(3)
            continue

        for index, current_screenshot in enumerate(current_screenshots):
            last_screenshot = last_screenshots[index]
            if is_similar(current_screenshot, last_screenshot):
                continue

            last_screenshots[index] = current_screenshot
            image = Image.fromarray(current_screenshot)
            timestamp = int(time.time())
            filename = f"{timestamp}_{index}.webp"
            image.save(
                os.path.join(screenshots_path, filename),
                format="webp",
                lossless=True,
            )

            if args.capture_only:
                continue

            ocr_result = extract_ocr_result_from_image(current_screenshot)
            if not ocr_result.text.strip():
                continue

            embedding: np.ndarray = get_embedding(ocr_result.text)
            normalization = normalize_ocr_text(ocr_result.text)
            active_app_name: str = get_active_app_name() or "Unknown App"
            active_window_title: str = get_active_window_title() or "Unknown Title"
            insert_entry(
                ocr_result.text,
                timestamp,
                embedding,
                active_app_name,
                active_window_title,
                captured_date=time.strftime("%Y-%m-%d", time.gmtime(timestamp)),
                normalized_text=normalization.normalized_text,
                normalization_version=normalization.version,
                normalization_issues_json=json.dumps(
                    normalization.issue_taxonomy,
                    sort_keys=True,
                ),
            )

        time.sleep(3)
