
import pydicom
import os

dicom_file = "test_dcm.dicom"

# Check if file exists
if not os.path.exists(dicom_file):
    print(f"❌ File '{dicom_file}' not found!")
    exit(1)

print(f"📂 Reading DICOM file: {dicom_file}")
print(f"📊 File size: {os.path.getsize(dicom_file)} bytes\n")

# Check file header
print("🔍 Checking file format...")
with open(dicom_file, 'rb') as f:
    header = f.read(132)
    print(f"First 4 bytes (hex): {header[:4].hex()}")
    print(f"First 4 bytes (text): {header[:4]}")
    
    # DICOM files should have 'DICM' at byte 128-131
    if len(header) >= 132:
        dicm_marker = header[128:132]
        print(f"DICM marker at 128-131: {dicm_marker}")
        if dicm_marker == b'DICM':
            print("✅ Valid DICOM header found!\n")
        else:
            print("⚠️ No 'DICM' marker found - might be implicit VR or corrupted\n")
    else:
        print("⚠️ File too small to contain DICOM header\n")

try:
    # Try reading with force=True to handle non-standard DICOM
    print("Attempting to read DICOM file...\n")
    ds = pydicom.dcmread(dicom_file, force=True)
    
    print("✅ DICOM file loaded successfully!\n")
    
    # Try to extract pixel data and convert to PNG
    if hasattr(ds, 'pixel_array'):
        import numpy as np
        from PIL import Image
        
        pixel_array = ds.pixel_array
        print(f"✅ Pixel data available: {pixel_array.shape}")
        print(f"   Data type: {pixel_array.dtype}")
        print(f"   Min value: {pixel_array.min()}, Max value: {pixel_array.max()}")
        
        # Normalize pixel data to 0-255 range
        pixel_normalized = pixel_array.astype(float)
        pixel_normalized = (pixel_normalized - pixel_normalized.min()) / (pixel_normalized.max() - pixel_normalized.min())
        pixel_normalized = (pixel_normalized * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if len(pixel_array.shape) == 2:
            # Grayscale image
            img = Image.fromarray(pixel_normalized, mode='L')
        elif len(pixel_array.shape) == 3:
            # RGB image
            img = Image.fromarray(pixel_normalized, mode='RGB')
        else:
            print(f"⚠️ Unsupported image shape: {pixel_array.shape}")
            img = None
        
        if img:
            output_file = dicom_file.replace('.dicom', '.png')
            img.save(output_file)
            print(f"\n✅ PNG image saved: {output_file}")
            print(f"   Image size: {img.size}")
            print(f"   Image mode: {img.mode}")
    else:
        print("⚠️ No pixel data found in DICOM file")
        
except Exception as e:
    print(f"❌ Error reading DICOM file: {type(e).__name__}")
    print(f"   Details: {str(e)}")
    import traceback
    traceback.print_exc()


