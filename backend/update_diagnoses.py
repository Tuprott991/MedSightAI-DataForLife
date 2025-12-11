"""
Update all diagnosis fields in cases table with random values
"""
import os
import sys
import random

# Add app to path
sys.path.append(os.path.dirname(__file__))

from app.config.database import SessionLocal
from app.models.models import Case


def update_all_diagnoses():
    """Update all cases with random diagnoses"""
    
    # Diagnosis pool
    diagnoses = [
        "Lung tumor",
        "Pneumonia",
        "Tuberculosis",
        "Other diseases",
        "No finding"
    ]
    
    print("="*80)
    print("🔄 Updating Case Diagnoses")
    print("="*80)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Get all cases
        cases = db.query(Case).all()
        total_cases = len(cases)
        
        print(f"\n📊 Found {total_cases} cases in database")
        
        if total_cases == 0:
            print("❌ No cases found")
            return
        
        # Confirm before proceeding
        print(f"\n📋 Diagnosis pool: {diagnoses}")
        response = input(f"\n⚠️  This will update {total_cases} cases with random diagnoses.\nContinue? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Cancelled")
            return
        
        # Update each case
        print("\n🔄 Updating cases...")
        updated_count = 0
        diagnosis_counts = {d: 0 for d in diagnoses}
        
        for idx, case in enumerate(cases, 1):
            # Randomly select diagnosis
            diagnosis = random.choice(diagnoses)
            
            # Update case
            case.diagnosis = diagnosis
            diagnosis_counts[diagnosis] += 1
            
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{total_cases} cases updated...")
            
            updated_count += 1
        
        # Commit all changes
        db.commit()
        
        print("\n" + "="*80)
        print(f"✅ Successfully updated {updated_count} cases")
        print("\n📊 Diagnosis Distribution:")
        for diagnosis, count in sorted(diagnosis_counts.items()):
            percentage = (count / total_cases * 100) if total_cases > 0 else 0
            print(f"   {diagnosis:20s}: {count:5d} cases ({percentage:5.1f}%)")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        db.rollback()
    
    finally:
        db.close()


if __name__ == "__main__":
    update_all_diagnoses()
