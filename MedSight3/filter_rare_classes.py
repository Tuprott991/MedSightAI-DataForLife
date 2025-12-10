"""
Filter out rare classes that have too few samples to learn.
This will improve model performance significantly.
"""

import pandas as pd
import argparse

# Classes with < 100 training samples (can't learn from them)
REMOVE_CONCEPTS = [
    'Edema',           # 13 train, 0 test
    'Clavicle fracture',  # 27 train, 2 test  
    'Lung cyst',       # 33 train, 2 test
    'Lung cavity',     # 51 train, 9 test
    'Emphysema',       # 81 train, 3 test
    'Rib fracture',    # 90 train, 11 test
]

REMOVE_TARGETS = [
    'COPD',  # 36 train, 2 test - impossible to learn
]

def filter_csv(input_csv, output_csv):
    """Remove rare classes from CSV."""
    print(f"\n📋 Processing: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"Original columns: {len(df.columns)}")
    print(f"Original samples: {len(df)}")
    
    # Drop rare columns
    cols_to_drop = REMOVE_CONCEPTS + REMOVE_TARGETS
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    if cols_to_drop:
        print(f"\n❌ Removing {len(cols_to_drop)} rare classes:")
        for col in cols_to_drop:
            if col in df.columns:
                pos_count = (df[col] == 1).sum()
                print(f"  - {col}: {pos_count} positive samples")
        
        df = df.drop(columns=cols_to_drop)
    
    # Save filtered CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved filtered CSV: {output_csv}")
    print(f"New columns: {len(df.columns)}")
    print(f"Removed: {len(cols_to_drop)} columns")
    
    # Show remaining concepts and targets
    meta_cols = ['image_id', 'rad_id']
    target_keywords = ['Lung tumor', 'Pneumonia', 'Tuberculosis', 'No finding', 'Other']
    
    remaining_cols = [c for c in df.columns if c not in meta_cols]
    target_cols = [c for c in remaining_cols if any(kw in c for kw in target_keywords)]
    concept_cols = [c for c in remaining_cols if c not in target_cols]
    
    print(f"\n📊 Remaining classes:")
    print(f"  Concepts: {len(concept_cols)}")
    print(f"  Targets: {len(target_cols)}")

def filter_bbox_csv(input_csv, output_csv):
    """Remove bboxes for rare classes."""
    print(f"\n📦 Processing bbox annotations: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"Original bbox rows: {len(df)}")
    
    # Remove rows with rare classes
    mask = ~df['class_name'].isin(REMOVE_CONCEPTS + REMOVE_TARGETS)
    df_filtered = df[mask]
    
    removed = len(df) - len(df_filtered)
    print(f"❌ Removed {removed} bbox rows for rare classes")
    
    # Save
    df_filtered.to_csv(output_csv, index=False)
    print(f"✅ Saved filtered bbox CSV: {output_csv}")
    print(f"Remaining bbox rows: {len(df_filtered)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--train_bbox_csv', type=str, required=True)
    parser.add_argument('--test_bbox_csv', type=str, required=True)
    parser.add_argument('--output_suffix', type=str, default='_filtered')
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 FILTERING RARE CLASSES")
    print("="*80)
    
    # Filter main CSVs
    train_out = args.train_csv.replace('.csv', f'{args.output_suffix}.csv')
    test_out = args.test_csv.replace('.csv', f'{args.output_suffix}.csv')
    
    filter_csv(args.train_csv, train_out)
    filter_csv(args.test_csv, test_out)
    
    # Filter bbox CSVs
    train_bbox_out = args.train_bbox_csv.replace('.csv', f'{args.output_suffix}.csv')
    test_bbox_out = args.test_bbox_csv.replace('.csv', f'{args.output_suffix}.csv')
    
    filter_bbox_csv(args.train_bbox_csv, train_bbox_out)
    filter_bbox_csv(args.test_bbox_csv, test_bbox_out)
    
    print("\n" + "="*80)
    print("✅ FILTERING COMPLETE!")
    print("="*80)
    print(f"\n📝 New files created:")
    print(f"  - {train_out}")
    print(f"  - {test_out}")
    print(f"  - {train_bbox_out}")
    print(f"  - {test_bbox_out}")
    print(f"\n💡 Use these filtered files for training!")

if __name__ == '__main__':
    main()
