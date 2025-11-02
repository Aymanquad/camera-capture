"""
Test script for the newly added functions in ocr_module.py and image_utils.py
"""
import cv2
import numpy as np
from ocr_module import extract_text_advanced, extract_business_card_text
from image_utils import find_document_quad_contour_enhanced, smart_enhance_for_ocr, estimate_noise_level

def create_test_image():
    """Create a simple test image with text"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Add some text-like shapes and patterns
    cv2.rectangle(img, (50, 50), (550, 150), (0, 0, 0), 2)  # Document border
    cv2.putText(img, "TEST DOCUMENT", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Email: test@example.com", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Phone: +1234567890", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Website: www.example.com", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img

def test_image_utils_functions():
    """Test new functions from image_utils.py"""
    print("=" * 60)
    print("TESTING image_utils.py FUNCTIONS")
    print("=" * 60)
    
    test_img = create_test_image()
    
    # Test 1: find_document_quad_contour_enhanced
    print("\n1. Testing find_document_quad_contour_enhanced()...")
    try:
        quad = find_document_quad_contour_enhanced(test_img)
        if quad is not None:
            print(f"   ✓ Found quadrilateral with {len(quad)} points")
            print(f"   Points: {quad}")
        else:
            print("   ✓ Function executed (no quad found - expected for synthetic image)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: smart_enhance_for_ocr
    print("\n2. Testing smart_enhance_for_ocr()...")
    try:
        enhanced_variants = smart_enhance_for_ocr(test_img)
        print(f"   ✓ Generated {len(enhanced_variants)} enhancement variants")
        for i, variant in enumerate(enhanced_variants):
            print(f"   Variant {i+1}: shape={variant.shape}, dtype={variant.dtype}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: estimate_noise_level
    print("\n3. Testing estimate_noise_level()...")
    try:
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        noise = estimate_noise_level(gray)
        print(f"   ✓ Estimated noise level: {noise:.2f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

def test_ocr_module_functions():
    """Test new functions from ocr_module.py"""
    print("\n" + "=" * 60)
    print("TESTING ocr_module.py FUNCTIONS")
    print("=" * 60)
    
    test_img = create_test_image()
    test_img_path = "test_image.jpg"
    cv2.imwrite(test_img_path, test_img)
    
    # Test 1: extract_text_advanced with general type
    print("\n1. Testing extract_text_advanced(doc_type='general')...")
    try:
        texts = extract_text_advanced(test_img, doc_type="general")
        print(f"   ✓ Extracted {len(texts)} text lines:")
        for i, text in enumerate(texts[:5], 1):  # Show first 5
            print(f"      {i}. {text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: extract_text_advanced with business_card type
    print("\n2. Testing extract_text_advanced(doc_type='business_card')...")
    try:
        texts = extract_text_advanced(test_img, doc_type="business_card")
        print(f"   ✓ Extracted {len(texts)} text lines:")
        for i, text in enumerate(texts[:5], 1):  # Show first 5
            print(f"      {i}. {text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: extract_text_advanced with pan_card type
    print("\n3. Testing extract_text_advanced(doc_type='pan_card')...")
    try:
        texts = extract_text_advanced(test_img, doc_type="pan_card")
        print(f"   ✓ Extracted {len(texts)} text lines:")
        for i, text in enumerate(texts[:5], 1):  # Show first 5
            print(f"      {i}. {text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: extract_business_card_text directly
    print("\n4. Testing extract_business_card_text() directly...")
    try:
        texts = extract_business_card_text(test_img)
        print(f"   ✓ Extracted {len(texts)} text lines:")
        for i, text in enumerate(texts[:5], 1):  # Show first 5
            print(f"      {i}. {text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 5: extract_text_advanced with file path
    print("\n5. Testing extract_text_advanced() with file path...")
    try:
        texts = extract_text_advanced(test_img_path, doc_type="general")
        print(f"   ✓ Extracted {len(texts)} text lines from file:")
        for i, text in enumerate(texts[:5], 1):  # Show first 5
            print(f"      {i}. {text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Cleanup
    import os
    if os.path.exists(test_img_path):
        os.remove(test_img_path)
        print(f"\n   Cleaned up test image: {test_img_path}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING NEW FUNCTIONS")
    print("=" * 60)
    
    test_image_utils_functions()
    test_ocr_module_functions()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60 + "\n")

