# Project Folder Structure

## Folders
- `camera-capture/`
  - `captures/`  
    _Storage for captured images_
  - `docs/`  
    _Documentation files_
  - `__pycache__/`  
    _Python artifact folder (auto-generated)_

## Core Files
- `camera-capture/`
  - `main.py`  
    _Integrates camera, OCR, storage_
  - `camera_module.py`  
    _Handles video stream, document detection, image capture_
  - `ocr_module.py`  
    _Handles OCR using EasyOCR or Google Vision_
  - `database_module.py`  
    _Handles SQLite/Postgres storage_
  - `requirements.txt`  
    _Python dependencies_
  - `README.md`  
    _Project overview_

## Optional/Future
- `celery_worker.py`  
  _For async jobs (optional)_
- `Dockerfile`  
  _For container deployment_
- `web/`  
  _Future: React/FastAPI dashboard/frontend_
- `tests/`  
  _Future: Unit tests_

---

## Suggestions & Advancements
- Consider separating config values into a `config.py` file (for DB paths, camera settings, etc).
- Place image-processing or utility functions in a `utils/` folder for scalabiIity.
- Prepare for environment variable usage (`.env`), especially for cloud services/API keys.
- If supporting multiple camera types, consider a camera abstraction class in a `core/` or `drivers/` folder.
- For web/desktop GUI (optional): add `web/` (FastAPI, React) or `desktop/` (PyQt/Electron) folders.
---

This structure is modular and allows for easy scaling into microservices or containerized deployment later.
