import pymongo
from pymongo import MongoClient
from datetime import datetime
import base64
import os
import cv2
import numpy as np
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB configuration from .env
MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'camera_capture')

# 3 Simple Collections
COLLECTION_CAPTURES = 'captures'    # Metadata only
COLLECTION_IMAGES = 'images'        # Original and processed images
COLLECTION_TEXTS = 'texts'          # Extracted text

_client: Optional[MongoClient] = None
_db = None
_captures_collection = None
_images_collection = None
_texts_collection = None


def set_mongodb_url(url: str):
    """Set MongoDB connection URL."""
    global MONGODB_URL
    MONGODB_URL = url


def connect_mongodb(url: Optional[str] = None) -> bool:
    """Connect to MongoDB and initialize 3 simple collections."""
    global _client, _db, _captures_collection, _images_collection, _texts_collection
    
    try:
        connection_url = url if url else MONGODB_URL
        _client = MongoClient(connection_url, serverSelectionTimeoutMS=5000)
        _client.server_info()  # Test connection
        
        _db = _client[DB_NAME]
        
        # Initialize 3 simple collections
        _captures_collection = _db[COLLECTION_CAPTURES]
        _images_collection = _db[COLLECTION_IMAGES]
        _texts_collection = _db[COLLECTION_TEXTS]
        
        # Create indexes for faster queries
        _captures_collection.create_index([("timestamp", -1)])
        _images_collection.create_index([("timestamp", -1)])
        _texts_collection.create_index([("timestamp", -1)])
        _texts_collection.create_index([("capture_id", 1)])
        
        print(f"Connected to MongoDB: {DB_NAME}")
        print(f"Collections: {COLLECTION_CAPTURES}, {COLLECTION_IMAGES}, {COLLECTION_TEXTS}")
        return True
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        _client = None
        _db = None
        _captures_collection = None
        _images_collection = None
        _texts_collection = None
        return False


def is_connected() -> bool:
    """Check if MongoDB is connected."""
    if _client is None:
        return False
    try:
        _client.server_info()
        return True
    except Exception:
        return False


def _image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image array to base64 string."""
    is_success, buffer = cv2.imencode('.jpg', image)
    if not is_success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def save_capture_to_db(
    original_image: np.ndarray,
    processed_image: Optional[np.ndarray] = None,
    text_content: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Save to 3 simple collections:
    1. captures - metadata
    2. images - original and processed images
    3. texts - extracted text
    
    Returns capture_id
    """
    if not is_connected():
        print("MongoDB not connected. Cannot save.")
        return None
    
    try:
        timestamp = datetime.now()
        
        # 1. Save to IMAGES collection
        image_doc = {
            'timestamp': timestamp,
            'original_image': _image_to_base64(original_image),
            'processed_image': _image_to_base64(processed_image) if processed_image is not None else None
        }
        image_result = _images_collection.insert_one(image_doc)
        image_id = str(image_result.inserted_id)
        
        # 2. Save to TEXTS collection
        text_lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        text_doc = {
            'timestamp': timestamp,
            'content': text_content,
            'lines': text_lines,
            'line_count': len(text_lines)
        }
        text_result = _texts_collection.insert_one(text_doc)
        text_id = str(text_result.inserted_id)
        
        # 3. Save to CAPTURES collection (metadata with references)
        capture_doc = {
            'timestamp': timestamp,
            'image_id': image_id,
            'text_id': text_id,
            'metadata': {
                'detection_method': metadata.get('detection_method', 'standard') if metadata else 'standard',
                'has_warped': metadata.get('has_warped', False) if metadata else False,
                'text_line_count': len(text_lines),
                'source': metadata.get('source', 'camera_capture_app') if metadata else 'camera_capture_app',
                **({} if not metadata else {k: v for k, v in metadata.items() 
                    if k not in ['detection_method', 'has_warped', 'source']})
            }
        }
        capture_result = _captures_collection.insert_one(capture_doc)
        capture_id = str(capture_result.inserted_id)
        
        # Update text with capture_id reference
        _texts_collection.update_one(
            {'_id': text_result.inserted_id},
            {'$set': {'capture_id': capture_id}}
        )
        
        print(f"Saved to MongoDB - Capture ID: {capture_id}")
        return capture_id
        
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_all_captures(limit: int = 100) -> list:
    """Get all captures with full data from all 3 collections."""
    if not is_connected():
        return []
    
    try:
        captures = list(_captures_collection.find().sort('timestamp', -1).limit(limit))
        
        # Join with images and texts
        for cap in captures:
            from bson import ObjectId
            
            # Get image data
            image = _images_collection.find_one({'_id': ObjectId(cap['image_id'])})
            if image:
                cap['original_image'] = image.get('original_image')
                cap['processed_image'] = image.get('processed_image')
            
            # Get text data
            text = _texts_collection.find_one({'_id': ObjectId(cap['text_id'])})
            if text:
                cap['text_content'] = text.get('content', '')
                cap['text_lines'] = text.get('lines', [])
        
        return captures
    except Exception as e:
        print(f"Error retrieving captures: {e}")
        return []


def get_capture_by_id(capture_id: str) -> Optional[Dict]:
    """Get a specific capture with full data."""
    if not is_connected():
        return None
    
    try:
        from bson import ObjectId
        capture = _captures_collection.find_one({'_id': ObjectId(capture_id)})
        if not capture:
            return None
        
        # Get image
        image = _images_collection.find_one({'_id': ObjectId(capture['image_id'])})
        
        # Get text
        text = _texts_collection.find_one({'_id': ObjectId(capture['text_id'])})
        
        return {
            'capture': capture,
            'image': image,
            'text': text
        }
    except Exception as e:
        print(f"Error retrieving capture: {e}")
        return None


def get_all_texts(limit: int = 100) -> list:
    """Get all extracted texts - easy way to read OCR results."""
    if not is_connected():
        return []
    
    try:
        texts = list(_texts_collection.find().sort('timestamp', -1).limit(limit))
        return texts
    except Exception as e:
        print(f"Error retrieving texts: {e}")
        return []


def close_connection():
    """Close MongoDB connection."""
    global _client, _db, _captures_collection, _images_collection, _texts_collection
    if _client:
        _client.close()
        _client = None
        _db = None
        _captures_collection = None
        _images_collection = None
        _texts_collection = None
        print("MongoDB connection closed")
