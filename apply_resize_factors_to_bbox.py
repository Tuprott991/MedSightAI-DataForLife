"""
Script to apply resize factors to bounding box coordinates in annotations.
This scales the original bounding box coordinates to match the resized 224x224 images.
"""

import pandas as pd
import numpy as np

def apply_resize_factors(annotations_path, resize_factors_path, output_path):
    """
    Apply resize factors to bounding box coordinates.
    
    Args:
        annotations_path: Path to annotations_train.csv
        resize_factors_path: Path to train_resize_factor.csv
        output_path: Path to save the updated annotations
    """
    # Read the CSV files
    print("Reading annotations...")
    annotations = pd.read_csv(annotations_path)
    print(f"Loaded {len(annotations)} annotations")
    
    print("Reading resize factors...")
    resize_factors = pd.read_csv(resize_factors_path)
    print(f"Loaded {len(resize_factors)} resize factors")
    
    # Merge annotations with resize factors on image_id
    print("Merging data...")
    merged = annotations.merge(resize_factors, on='image_id', how='left')
    
    # Check for missing resize factors
    missing_factors = merged['resize_factor_h'].isna().sum()
    if missing_factors > 0:
        print(f"Warning: {missing_factors} annotations have missing resize factors")
    
    # Apply resize factors to bounding box coordinates
    # Only update rows where bbox coordinates exist (not NaN)
    print("Applying resize factors to bounding boxes...")
    
    # Create mask for rows with valid bounding boxes
    has_bbox = merged['x_min'].notna()
    
    # Apply resize factors to x coordinates (width)
    merged.loc[has_bbox, 'x_min_resized'] = merged.loc[has_bbox, 'x_min'] * merged.loc[has_bbox, 'resize_factor_w']
    merged.loc[has_bbox, 'x_max_resized'] = merged.loc[has_bbox, 'x_max'] * merged.loc[has_bbox, 'resize_factor_w']
    
    # Apply resize factors to y coordinates (height)
    merged.loc[has_bbox, 'y_min_resized'] = merged.loc[has_bbox, 'y_min'] * merged.loc[has_bbox, 'resize_factor_h']
    merged.loc[has_bbox, 'y_max_resized'] = merged.loc[has_bbox, 'y_max'] * merged.loc[has_bbox, 'resize_factor_h']
    
    # Replace original bbox coordinates with resized ones
    merged.loc[has_bbox, 'x_min'] = merged.loc[has_bbox, 'x_min_resized']
    merged.loc[has_bbox, 'x_max'] = merged.loc[has_bbox, 'x_max_resized']
    merged.loc[has_bbox, 'y_min'] = merged.loc[has_bbox, 'y_min_resized']
    merged.loc[has_bbox, 'y_max'] = merged.loc[has_bbox, 'y_max_resized']
    
    # Select only the original annotation columns (same structure as input)
    # Dynamically detect which columns were in the original annotations
    original_columns = list(annotations.columns)
    result = merged[original_columns]
    
    # Save to CSV
    print(f"Saving results to {output_path}...")
    result.to_csv(output_path, index=False)
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total annotations: {len(result)}")
    print(f"Annotations with bounding boxes: {has_bbox.sum()}")
    print(f"Annotations without bounding boxes (No finding): {(~has_bbox).sum()}")
    
    # Show sample of resized bounding boxes
    print("\n=== Sample of resized bounding boxes ===")
    sample = result[result['x_min'].notna()].head(3)
    print(sample[['image_id', 'class_name', 'x_min', 'y_min', 'x_max', 'y_max']])
    
    print("\n✓ Done! Updated annotations saved.")
    
    return result


if __name__ == "__main__":
    # Define paths
    annotations_path = "annotations_test.csv"
    resize_factors_path = "test_resize_factor.csv"
    output_path = "annotations_test_resized.csv"
    
    # Apply resize factors
    result = apply_resize_factors(annotations_path, resize_factors_path, output_path)
