# MongoDB Integration Setup

## Folder Structure
The app now organizes files into separate folders:
- `captures/` - Original captured images
- `captures/processed/` - Processed/enhanced images
- `captures/text/` - Extracted OCR text files

## MongoDB Configuration

### Option 1: Environment Variable (Recommended)
Set the MongoDB URL as an environment variable:
```bash
# Windows PowerShell
$env:MONGODB_URL="mongodb+srv://username:password@cluster.mongodb.net/"

# Windows CMD
set MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/

# Linux/Mac
export MONGODB_URL="mongodb+srv://username:password@cluster.mongodb.net/"
```

### Option 2: Modify database_module.py
Edit the `MONGODB_URL` variable directly in `database_module.py`:
```python
MONGODB_URL = 'mongodb+srv://username:password@cluster.mongodb.net/'
```

## MongoDB Document Structure
Each capture is stored as a document with:
- `timestamp`: DateTime of capture
- `original_image`: Base64-encoded original image
- `processed_image`: Base64-encoded processed image (if available)
- `text_content`: Full extracted text
- `text_lines`: Array of text lines
- `file_paths`: Object with original and processed filenames
- `metadata`: Additional info (detection method, text count, etc.)

## Database Name
Default database: `camera_capture`
Default collection: `captures`

You can modify these in `database_module.py`:
```python
DB_NAME = 'your_database_name'
COLLECTION_NAME = 'your_collection_name'
```

