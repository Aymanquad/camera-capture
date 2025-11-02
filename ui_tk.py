import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from datetime import datetime
import time
import threading
from collections import deque
from ocr_module import extract_text_robust, extract_text_advanced
from image_utils import find_document_quad_contour, find_document_quad_contour_enhanced, warp_perspective_to_topdown, enhance_for_ocr, smart_enhance_for_ocr, rotate_image, generate_horizontal_strips
from database_module import connect_mongodb, save_capture_to_db, is_connected


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Camera - Detect Text')
        self.video_label = ttk.Label(root)
        self.video_label.pack(padx=8, pady=8)

        buttons = ttk.Frame(root)
        buttons.pack(pady=(0, 8))
        self.detect_btn = ttk.Button(buttons, text='Detect', command=self.on_detect)
        self.detect_btn.grid(row=0, column=0, padx=6)
        self.quit_btn = ttk.Button(buttons, text='Quit', command=self.on_quit)
        self.quit_btn.grid(row=0, column=1, padx=6)

        self.status_var = tk.StringVar(value='Ready')
        self.status_label = ttk.Label(root, textvariable=self.status_var)
        self.status_label.pack(pady=(0, 4))

        self.output = tk.Text(root, height=12, wrap='word')
        self.output.configure(state='disabled')
        self.output.pack(fill='both', expand=False, padx=8, pady=(0, 8))

        self.cap = cv2.VideoCapture(0)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.latest_frame = None
        self.frame_buffer = deque(maxlen=90)
        self.buffer_lock = threading.Lock()
        self.detecting = False
        
        # Initialize MongoDB connection
        if connect_mongodb():
            self.status_var.set('Ready (DB Connected)')
        else:
            self.status_var.set('Ready (DB Not Connected)')

        self.update_video()

    def update_video(self):
        if self.cap.isOpened():
            self.cap.grab()
            ret, frame = self.cap.read()
            if ret:
                self.cap.grab()
                ret2, frame2 = self.cap.read()
                if ret2:
                    frame = frame2
                self.latest_frame = frame
                with self.buffer_lock:
                    self.frame_buffer.append(frame.copy())
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        self.root.after(20, self.update_video)

    def _set_output_text(self, text: str):
        self.output.configure(state='normal')
        self.output.delete('1.0', tk.END)
        self.output.insert(tk.END, text)
        self.output.configure(state='disabled')

    def on_detect(self):
        if self.detecting:
            return
        self.detecting = True
        self.detect_btn.state(['disabled'])
        self.status_var.set('Capturing (1s)... Hold steady')
        self.status_label.configure(foreground='blue')
        t = threading.Thread(target=self._detect_worker, daemon=True)
        t.start()

    def _detect_worker(self):
        # Sample frames from buffer for ~1s and pick the sharpest
        start = time.time()
        best_frame = None
        best_var = -1.0
        while time.time() - start < 1.0:
            with self.buffer_lock:
                frames = list(self.frame_buffer)[-5:]
            for f in frames:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                focus = cv2.Laplacian(gray, cv2.CV_64F).var()
                if focus > best_var:
                    best_var = focus
                    best_frame = f
            time.sleep(0.03)

        if best_frame is None:
            self.root.after(0, self._finish_detect, None, None)
            return

        def update_status(msg):
            self.root.after(0, lambda: self.status_var.set(msg))

        update_status('Processing image...')

        # Document warp and enhancement - try enhanced detection first
        warped_img = None
        try:
            quad = find_document_quad_contour_enhanced(best_frame)
            if quad is None:
                # Fallback to original method
                quad = find_document_quad_contour(best_frame)
            if quad is not None:
                try:
                    warped_img = warp_perspective_to_topdown(best_frame, quad)
                    update_status('Document detected, warping...')
                except Exception:
                    warped_img = None
        except Exception:
            warped_img = None
        
        base_for_enhance = warped_img if warped_img is not None else best_frame
        
        # Use smart enhancement (returns multiple variants) - use only first best variant
        best_enhanced = None
        try:
            update_status('Enhancing image...')
            smart_enhanced = smart_enhance_for_ocr(base_for_enhance)
            if smart_enhanced:
                best_enhanced = smart_enhanced[0]  # Use only the first (best) variant
        except Exception:
            pass
        
        # Fallback to original enhancement
        if best_enhanced is None:
            try:
                best_enhanced = enhance_for_ocr(base_for_enhance)
            except Exception:
                best_enhanced = None

        # OCR using advanced and robust functions - optimized to reduce calls
        texts = []
        update_status('Running OCR... (this may take a moment)')
        
        # Primary OCR on best image variant
        try:
            if warped_img is not None:
                texts.extend(extract_text_advanced(warped_img, doc_type="general"))
            else:
                texts.extend(extract_text_advanced(best_frame, doc_type="general"))
        except Exception:
            pass
        
        # Enhanced variant OCR (only if we have one)
        if best_enhanced is not None:
            try:
                proc_bgr = cv2.cvtColor(best_enhanced, cv2.COLOR_GRAY2BGR)
                texts.extend(extract_text_robust(proc_bgr, min_conf=0.5, upscale=1.5))
            except Exception:
                pass

        # One rotation variant to catch skew (reduced from 2)
        try:
            rotated = rotate_image(base_for_enhance, -3.0)
            texts.extend(extract_text_robust(rotated, min_conf=0.55, upscale=1.5))
        except Exception:
            pass

        # Dedup preserve order
        seen = set()
        merged = []
        for t in texts:
            s = t.strip()
            if s and s not in seen:
                seen.add(s)
                merged.append(s)

        # Prepare data for MongoDB
        full_text = "\n".join(merged)
        
        # Convert processed image from grayscale to BGR if needed
        processed_image_bgr = None
        if best_enhanced is not None:
            # If grayscale, convert to BGR (3 channels)
            if len(best_enhanced.shape) == 2:
                processed_image_bgr = cv2.cvtColor(best_enhanced, cv2.COLOR_GRAY2BGR)
            else:
                processed_image_bgr = best_enhanced
        
        # Save to MongoDB (no local files)
        update_status('Saving to database...')
        metadata = {
            'detection_method': 'enhanced' if warped_img is not None else 'standard',
            'has_warped': warped_img is not None,
            'text_line_count': len(merged),
            'source': 'camera_capture_app'
        }
        
        db_id = save_capture_to_db(
            original_image=best_frame,
            processed_image=processed_image_bgr,
            text_content=full_text,
            metadata=metadata
        )
        
        if db_id:
            update_status(f'Saved to database (ID: {db_id[:8]}...)')
        else:
            update_status('Warning: Could not save to database')

        self.root.after(0, self._finish_detect, db_id, full_text)

    def _finish_detect(self, db_id, full_text):
        if db_id is None:
            self.status_var.set('ERROR: Could not capture frame or save to database')
            self.status_label.configure(foreground='red')
        else:
            if full_text:
                self._set_output_text(full_text)
                msg = f'SUCCESS: Saved to MongoDB (ID: {db_id[:12]}...)'
                self.status_var.set(msg)
                self.status_label.configure(foreground='green')
                self.root.clipboard_clear()
                self.root.clipboard_append(full_text)
            else:
                self._set_output_text('')
                self.status_var.set('WARNING: Saved but no text detected')
                self.status_label.configure(foreground='orange')
        self.detect_btn.state(['!disabled'])
        self.detecting = False

    def on_quit(self):
        if self.cap:
            self.cap.release()
        from database_module import close_connection
        close_connection()
        self.root.destroy()


def run():
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
