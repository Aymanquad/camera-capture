# MongoDB Schema - 3 Simple Collections

## Database: `camera_capture`

### Collection 1: `captures`
**Purpose:** Store metadata and references

```javascript
{
  _id: ObjectId,
  timestamp: ISODate,
  image_id: ObjectId,        // Reference to images collection
  text_id: ObjectId,          // Reference to texts collection
  metadata: {
    detection_method: String,
    has_warped: Boolean,
    text_line_count: Number,
    source: String
  }
}
```

---

### Collection 2: `images`
**Purpose:** Store original and processed images as base64

```javascript
{
  _id: ObjectId,
  timestamp: ISODate,
  original_image: String,     // Base64 encoded JPEG
  processed_image: String      // Base64 encoded JPEG (or null)
}
```

---

### Collection 3: `texts`
**Purpose:** Store extracted OCR text - EASY TO READ!

```javascript
{
  _id: ObjectId,
  timestamp: ISODate,
  capture_id: String,          // Reference back to capture
  content: String,             // Full extracted text
  lines: [String],             // Array of text lines
  line_count: Number
}
```

---

## How to Read Extracted Text:

### Option 1: Query texts collection directly
```javascript
db.texts.find().sort({timestamp: -1})
```

### Option 2: Use helper function in Python
```python
from database_module import get_all_texts
texts = get_all_texts(limit=10)
for text in texts:
    print(text['content'])  # Full text
    print(text['lines'])    # Array of lines
```

### Option 3: Get full capture with all data
```python
from database_module import get_all_captures
captures = get_all_captures(limit=10)
for cap in captures:
    print(cap['text_content'])  # Extracted text
    print(cap['text_lines'])      # Lines array
```

---

## Simple Structure:
- **captures** → links everything together
- **images** → stores images
- **texts** → stores text (easy to read!)
