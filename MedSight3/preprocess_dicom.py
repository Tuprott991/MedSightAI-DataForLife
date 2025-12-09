"""
Script để preprocess DICOM thành PNG/JPG nhanh nhất có thể.
Sử dụng multiprocessing để xử lý song song trên nhiều CPU cores.

Usage:
    python preprocess_dicom.py --input_dir train/ --output_dir train_png/ --num_workers 8
"""

import os
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import pydicom
import numpy as np
import cv2

def process_single_dicom(args):
    """
    Xử lý 1 file DICOM thành PNG.
    Args: tuple (dicom_path, output_path, target_size)
    """
    dicom_path, output_path, target_size = args
    
    try:
        # Kiểm tra nếu output đã tồn tại → skip
        if os.path.exists(output_path):
            return True, None
        
        # Đọc DICOM file
        dicom = pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array
        
        # Photometric Interpretation handling (quan trọng cho X-Ray)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array
            
        # Normalize về 0-255
        pixel_array = pixel_array.astype(np.float32)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0
        pixel_array = pixel_array.astype(np.uint8)
        
        # Resize
        img = cv2.resize(pixel_array, target_size)
        
        # Save PNG
        cv2.imwrite(output_path, img)
        
        return True, None
    except Exception as e:
        return False, f"{dicom_path}: {str(e)}"

def preprocess_dataset(input_dir, output_dir, csv_file=None, num_workers=None, format='png', size=224):
    """
    Preprocess toàn bộ dataset DICOM thành PNG/JPG.
    
    Args:
        input_dir: Folder chứa DICOM files
        output_dir: Folder output cho PNG/JPG
        csv_file: File CSV chứa danh sách image_id (optional, nếu None sẽ process tất cả)
        num_workers: Số CPU cores dùng (None = auto detect)
        format: 'png' hoặc 'jpg'
        size: Target size for resizing (default: 224)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Auto detect số workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # Để lại 2 cores cho system
    
    print(f"🚀 Starting preprocessing with {num_workers} workers...")
    print(f"📂 Input: {input_dir}")
    print(f"📂 Output: {output_dir}")
    print(f"🎨 Format: {format.upper()}")
    print(f"📐 Target size: {size}x{size}")
    
    # Lấy danh sách files cần xử lý
    if csv_file and os.path.exists(csv_file):
        # Nếu có CSV, chỉ process các ảnh trong CSV
        df = pd.read_csv(csv_file)
        image_ids = df['image_id'].unique()
        print(f"📋 Found {len(image_ids)} unique images in CSV")
    else:
        # Nếu không có CSV, process tất cả files trong folder
        image_ids = [f.stem for f in Path(input_dir).glob('*.dicom')]
        if len(image_ids) == 0:
            # Try without extension
            image_ids = [f.stem for f in Path(input_dir).iterdir() if f.is_file()]
        print(f"📋 Found {len(image_ids)} DICOM files in directory")
    
    # Chuẩn bị tasks cho multiprocessing
    tasks = []
    for img_id in image_ids:
        # Tìm file DICOM (có thể có hoặc không có extension)
        dicom_path = Path(input_dir) / f"{img_id}.dicom"
        if not dicom_path.exists():
            dicom_path = Path(input_dir) / img_id
        
        if not dicom_path.exists():
            print(f"⚠️  Skip {img_id}: file not found")
            continue
        
        output_path = Path(output_dir) / f"{img_id}.{format}"
        tasks.append((str(dicom_path), str(output_path), (size, size)))
    
    print(f"📦 Total tasks: {len(tasks)}")
    
    # Process với multiprocessing + progress bar
    success_count = 0
    error_count = 0
    errors = []
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_dicom, tasks),
            total=len(tasks),
            desc="Processing",
            unit="images"
        ))
    
    # Đếm kết quả
    for success, error_msg in results:
        if success:
            success_count += 1
        else:
            error_count += 1
            if error_msg:
                errors.append(error_msg)
    
    # In kết quả
    print(f"\n✅ Successfully processed: {success_count}/{len(tasks)}")
    if error_count > 0:
        print(f"❌ Errors: {error_count}")
        print("\nFirst 10 errors:")
        for error in errors[:10]:
            print(f"  - {error}")
    
    print(f"\n🎉 Done! Output saved to: {output_dir}")
    
    # Tính dung lượng tiết kiệm
    if success_count > 0:
        sample_dicom = Path(tasks[0][0])
        sample_png = Path(tasks[0][1])
        if sample_dicom.exists() and sample_png.exists():
            dicom_size = sample_dicom.stat().st_size / 1024 / 1024  # MB
            png_size = sample_png.stat().st_size / 1024 / 1024
            ratio = (dicom_size / png_size) if png_size > 0 else 0
            print(f"📊 Size comparison (1 sample):")
            print(f"   DICOM: {dicom_size:.2f} MB")
            print(f"   {format.upper()}: {png_size:.2f} MB")
            print(f"   Ratio: {ratio:.1f}x")

def main():
    parser = argparse.ArgumentParser(description='Preprocess DICOM to PNG/JPG for faster training')
    
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Input directory containing DICOM files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for PNG/JPG files')
    parser.add_argument('--csv_file', type=str, default=None,
                        help='CSV file with image_id column (optional)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of CPU workers (default: auto)')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpg'],
                        help='Output format: png or jpg (default: png)')
    parser.add_argument('--size', type=int, default=224,
                        help='Target size for resizing (default: 224)')
    
    args = parser.parse_args()
    
    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_file=args.csv_file,
        num_workers=args.num_workers,
        format=args.format,
        size=args.size
    )

if __name__ == '__main__':
    main()
