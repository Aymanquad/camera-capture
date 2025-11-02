import cv2
import os
from datetime import datetime

# Windows-only non-blocking keyboard (kept for potential fallback)
try:
    import msvcrt  # type: ignore
    HAS_MSVCRT = True
except Exception:
    HAS_MSVCRT = False

CAPTURE_DIR = 'captures'
ASCII_CHARS = "@%#*+=-:. "  # dark -> light


def ensure_capture_dir():
    if not os.path.exists(CAPTURE_DIR):
        os.makedirs(CAPTURE_DIR)


def _save_frame(frame):
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = os.path.join(CAPTURE_DIR, filename)
    cv2.imwrite(img_path, frame)
    print(f"Captured {img_path}")
    return img_path


def _frame_to_ascii(frame, cols=80):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w == 0 or h == 0:
        return ""
    scale = cols / float(w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale * 0.5))
    small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    indices = (small / 256.0 * (len(ASCII_CHARS) - 1)).astype('uint8')
    lines = ["".join(ASCII_CHARS[idx] for idx in row) for row in indices]
    return "\n".join(lines)


# Preferred live window mode
def run_live_camera_with_detect(on_detect):
    """
    Shows a live camera window. Keys:
      d: capture current frame, save, and call on_detect(image_path)
      q: quit
    """
    ensure_capture_dir()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Reduce buffering to minimize latency
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    window_name = 'Camera - press d=detect, q=quit'
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        print("GUI not available for OpenCV windowing.")
        cap.release()
        raise

    try:
        while True:
            # Grab+retrieve twice to flush buffer and get freshest frame
            cap.grab()
            ret, frame = cap.read()
            if not ret:
                continue
            cap.grab()
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('d'):
                img_path = _save_frame(frame)
                try:
                    on_detect(img_path)
                except Exception as _:
                    pass
            elif key == ord('q'):
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


# Fallback CLI loop retained (not used by default)
def interactive_capture_loop():
    ensure_capture_dir()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    latest_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _ = cap.read()
        latest_frame = frame
        if HAS_MSVCRT and msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                key = ch.decode('utf-8').lower()
            except Exception:
                key = ''
            if key == 'd' and latest_frame is not None:
                img_path = _save_frame(latest_frame)
                cap.release()
                return img_path
            elif key == 'q':
                break
    cap.release()
    return None
