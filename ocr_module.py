import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*pin_memory.*",
    category=UserWarning,
    module=r"torch\.utils\.data\.dataloader"
)

import easyocr
import cv2
import numpy as np
import re
from image_utils import smart_enhance_for_ocr

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'], gpu=False)


def _ensure_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def extract_text_from_image(image_input):
    """
    Accepts either a file path (str) or a numpy image (BGR).
    Returns list of unique lines with paragraph grouping.
    """
    if isinstance(image_input, str):
        result = reader.readtext(image_input, detail=0, paragraph=True)
    else:
        rgb = _ensure_rgb(image_input)
        result = reader.readtext(rgb, detail=0, paragraph=True)

    # Deduplicate while preserving order
    seen = set()
    unique_lines = []
    for line in result:
        if not line:
            continue
        norm = line.strip()
        if norm and norm not in seen:
            seen.add(norm)
            unique_lines.append(norm)
    return unique_lines


def extract_text_robust(image_input, min_conf: float = 0.5, upscale: float = 1.5):
    """
    More accurate OCR: upscales input and filters low-confidence results.
    Returns list of unique lines (ordered top-to-bottom).
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input
    if img is None:
        return []

    # Upscale to help small text
    if upscale and upscale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC)

    rgb = _ensure_rgb(img)
    results = reader.readtext(rgb, detail=1, paragraph=False)

    # results: list of (bbox, text, conf)
    items = []
    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        bbox, text, conf = item
        if text and conf is not None and conf >= min_conf:
            # y-position for ordering
            ys = [p[1] for p in bbox]
            y_mean = float(sum(ys) / len(ys)) if ys else 0.0
            items.append((y_mean, text.strip()))

    items.sort(key=lambda x: x[0])
    seen = set()
    unique_lines = []
    for _, t in items:
        if t and t not in seen:
            seen.add(t)
            unique_lines.append(t)
    return unique_lines











#i added this , try it out :
def extract_text_advanced(image_input, doc_type="general"):
    """
    Advanced OCR with document-type specific preprocessing
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input.copy()
    
    # Document type specific handling
    if doc_type == "business_card":
        return extract_business_card_text(img)
    elif doc_type == "pan_card":
        # Use robust extraction for PAN cards
        return extract_text_robust(img, min_conf=0.6, upscale=2.0)
    else:
        # Use robust extraction for general text
        return extract_text_robust(img, min_conf=0.5, upscale=1.5)

def extract_business_card_text(img):
    """Specialized processing for business cards"""
    # Business cards often have small text and glossy surfaces
    enhanced_variants = smart_enhance_for_ocr(img)
    
    all_texts = []
    for variant in enhanced_variants:
        # Convert back to BGR for EasyOCR
        variant_bgr = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
        
        # Try different scales for business card text
        for scale in [1.5, 2.0, 2.5]:
            texts = extract_text_robust(variant_bgr, min_conf=0.4, upscale=scale)
            all_texts.extend(texts)
    
    return post_process_business_card_text(all_texts)

def post_process_business_card_text(texts):
    """Post-process business card specific text patterns"""
    seen = set()
    unique_lines = []
    
    for text in texts:
        cleaned = text.strip()
        if not cleaned or len(cleaned) < 2:
            continue
            
        # Business card specific filtering
        if is_likely_contact_info(cleaned):
            if cleaned not in seen:
                seen.add(cleaned)
                unique_lines.append(cleaned)
    
    return unique_lines

def is_likely_contact_info(text):
    """Heuristic to identify contact information"""
    # Email pattern
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', text):
        return True
    # Phone number pattern
    if re.match(r'^[\+]?[0-9\s\-\(\)]{7,15}$', text.replace(' ', '')):
        return True
    # Website pattern
    if re.match(r'^(https?://)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
        return True
    return len(text) > 3  # General text of reasonable length


