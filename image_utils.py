import cv2
import numpy as np


def order_points(pts):
    # pts: 4x2 array
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def find_document_quad_contour(image):
    # Return 4-point contour of the largest quadrilateral, or None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None




#i added this , try it out :
def find_document_quad_contour_enhanced(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multiple preprocessing techniques
    # Option 1: Adaptive threshold for varying lighting
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Option 2: Otsu's threshold
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Option 3: Morphological gradient for edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
    # Combine multiple edge detection methods
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(thresh1, 50, 150)
    
    combined_edges = cv2.bitwise_or(edges1, edges2)
    combined_edges = cv2.dilate(combined_edges, None, iterations=2)
    combined_edges = cv2.erode(combined_edges, None, iterations=1)
    
    contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # Additional validation: check if quadrilateral is reasonably rectangular
            area = cv2.contourArea(approx)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if area > 1000 and 0.3 < aspect_ratio < 3.0:
                return approx.reshape(4, 2)
    return None









def warp_perspective_to_topdown(image, quad_pts):
    rect = order_points(quad_pts.astype('float32'))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def enhance_for_ocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Bilateral filter to reduce noise while keeping edges
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    # Unsharp masking to sharpen text edges
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(sharp)
    # Adaptive threshold
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    # Morphology open to clean speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened




#i added this , try it out :
def estimate_noise_level(gray_image):
    """Estimate noise level in image"""
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return np.var(laplacian)

def smart_enhance_for_ocr(image_bgr):
    """Adaptive enhancement based on image characteristics"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Analyze image quality
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Dynamic parameter adjustment
    if brightness < 50:  # Dark image
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    elif brightness > 200:  # Overexposed
        gray = cv2.convertScaleAbs(gray, alpha=0.7, beta=0)
    
    # Adaptive filtering based on noise level
    noise_level = estimate_noise_level(gray)
    if noise_level > 10:
        gray = cv2.bilateralFilter(gray, 5, 75, 75)
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Multiple enhancement strategies
    enhanced_images = []
    
    # Strategy 1: CLAHE + Adaptive Threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    th1 = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 21, 10)
    enhanced_images.append(th1)
    
    # Strategy 2: Morphological operations for glossy surfaces
    kernel = np.ones((1,1), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    th2 = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 15, 12)
    enhanced_images.append(th2)
    
    # Strategy 3: For reflective surfaces - try different thresholding
    _, th3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_images.append(th3)
    
    return enhanced_images  # Return multiple variants for OCR









def rotate_image(image_bgr, angle_degrees: float):
    h, w = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def generate_horizontal_strips(image_bgr, num_strips: int = 4, overlap: float = 0.15):
    h, w = image_bgr.shape[:2]
    strips = []
    base = int(h / num_strips)
    step = int(base * (1 - overlap)) if overlap < 1 else base
    if step <= 0:
        step = max(1, base)
    y = 0
    while y < h:
        y2 = min(h, y + base)
        strip = image_bgr[y:y2, 0:w]
        if strip.size > 0:
            strips.append(strip)
        y += step
    return strips
