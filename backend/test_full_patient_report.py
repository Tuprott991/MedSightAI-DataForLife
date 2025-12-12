"""
Test Full Patient Report API Endpoint
This script demonstrates how to use the /api/v1/reports/full/{case_id} endpoint
to get comprehensive patient information with all related data
"""
import requests
import json

# Configuration
BACKEND_URL = "http://localhost:8000"
CASE_ID = "00d84925-ec50-416f-b180-07b48fa81826"  # Replace with actual case UUID


def test_full_patient_report():
    """
    Test the full patient report endpoint
    
    This endpoint returns:
    - Patient information (name, age, gender, blood type, status, underlying conditions)
    - Case details (image paths, diagnosis, findings, timestamp)
    - AI analysis results (predicted diagnosis, bounding boxes, concepts)
    - Generated medical report (model report, doctor report, feedback)
    """
    print("="*80)
    print("📋 Testing Full Patient Report API")
    print("="*80)
    
    # Endpoint URL
    url = f"{BACKEND_URL}/api/v1/reports/full/{CASE_ID}"
    
    print(f"\n📡 Calling: GET {url}")
    
    try:
        # Make request
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        print("\n✅ Success! Full Patient Report Retrieved:")
        print(json.dumps(result, indent=2, default=str))
        
        # Display organized summary
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE PATIENT REPORT")
        print("="*80)
        
        # Patient Information
        print("\n👤 PATIENT INFORMATION")
        print("-" * 40)
        print(f"   Name: {result.get('patient_name')}")
        print(f"   Age: {result.get('patient_age')} years old")
        print(f"   Gender: {result.get('patient_gender')}")
        print(f"   Blood Type: {result.get('blood_type')}")
        print(f"   Status: {result.get('status')}")
        print(f"   Phone: {result.get('phone_number')}")
        
        # Underlying Conditions
        underlying = result.get('underlying_condition', {})
        if underlying:
            conditions = [k for k, v in underlying.items() if v is True]
            if conditions:
                print(f"   Underlying Conditions: {', '.join(conditions)}")
            else:
                print(f"   Underlying Conditions: None")
        
        # Case Information
        print("\n🏥 CASE INFORMATION")
        print("-" * 40)
        print(f"   Case ID: {result.get('case_id')}")
        print(f"   Timestamp: {result.get('case_timestamp')}")
        print(f"   Diagnosis: {result.get('diagnosis')}")
        print(f"   Findings: {result.get('findings')}")
        print(f"   Image URL: {result.get('image_path')}")
        if result.get('processed_img_path'):
            print(f"   DICOM URL: {result.get('processed_img_path')}")
        
        # AI Analysis Results
        if result.get('ai_result_id'):
            print("\n🤖 AI ANALYSIS RESULTS")
            print("-" * 40)
            print(f"   AI Result ID: {result.get('ai_result_id')}")
            print(f"   Predicted Diagnosis: {result.get('predicted_diagnosis')}")
            
            concepts = result.get('concepts', {})
            detected = concepts.get('detected_concepts', [])
            if detected:
                print(f"   Detected Abnormalities: {', '.join(detected)}")
                
                # Bounding Boxes
                bbox_data = result.get('bounding_box', {})
                detections = bbox_data.get('detections', [])
                if detections:
                    print(f"\n   Bounding Boxes ({len(detections)} detections):")
                    for i, det in enumerate(detections, 1):
                        print(f"      {i}. {det.get('concept')}")
                        print(f"         Probability: {det.get('probability'):.4f}")
                        bbox = det.get('bbox', [])
                        if bbox:
                            print(f"         Location: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            else:
                print(f"   No abnormalities detected")
            
            print(f"   AI Analysis Time: {result.get('ai_result_created_at')}")
        else:
            print("\n🤖 AI ANALYSIS RESULTS")
            print("-" * 40)
            print(f"   ⚠️  AI analysis not performed yet")
            print(f"   Run CAM inference first: POST /api/v1/analysis/cam-inference/{CASE_ID}")
        
        # Medical Report
        if result.get('report_id'):
            print("\n📄 MEDICAL REPORT")
            print("-" * 40)
            print(f"   Report ID: {result.get('report_id')}")
            
            model_report = result.get('model_report', {})
            if model_report:
                print(f"\n   MeSH Terms: {model_report.get('MeSH')}")
                print(f"   Image Type: {model_report.get('Image')}")
                print(f"   Indication: {model_report.get('Indication')}")
                print(f"   Comparison: {model_report.get('Comparison')}")
                
                print(f"\n   FINDINGS:")
                findings_text = model_report.get('Findings', '')
                # Wrap text at 70 characters
                import textwrap
                wrapped = textwrap.fill(findings_text, width=70, initial_indent='      ', subsequent_indent='      ')
                print(wrapped)
                
                print(f"\n   IMPRESSION:")
                impression_text = model_report.get('Impression', '')
                wrapped = textwrap.fill(impression_text, width=70, initial_indent='      ', subsequent_indent='      ')
                print(wrapped)
            
            if result.get('doctor_report'):
                print(f"\n   DOCTOR'S REPORT:")
                print(f"      {result.get('doctor_report')}")
            
            if result.get('feedback_note'):
                print(f"\n   FEEDBACK:")
                print(f"      {result.get('feedback_note')}")
            
            print(f"\n   Report Generated: {result.get('report_created_at')}")
        else:
            print("\n📄 MEDICAL REPORT")
            print("-" * 40)
            print(f"   ⚠️  Medical report not generated yet")
            print(f"   Generate report: POST /api/v1/reports/generate")
        
        print("\n" + "="*80)
        
        return result
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def list_available_cases():
    """
    Helper function to list available cases
    """
    url = f"{BACKEND_URL}/api/v1/cases/"
    params = {"page": 1, "page_size": 10}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print("📋 Available Cases:")
        for case in data.get('cases', []):
            print(f"   - {case['id']} (Patient: {case['patient_id']}, Diagnosis: {case.get('diagnosis', 'N/A')})")
        
        return data.get('cases', [])
        
    except Exception as e:
        print(f"❌ Error fetching cases: {e}")
        return []


if __name__ == "__main__":
    print("Checking available cases...\n")
    cases = list_available_cases()
    
    if cases and CASE_ID == "01e58e5f-1c29-47c3-8b57-df985b2433c8":
        print(f"\n💡 Update CASE_ID variable with one of the above case IDs")
        print(f"   Example: CASE_ID = '{cases[0]['id']}'")
    
    print("\n")
    test_full_patient_report()


"""
Example Response Structure:
{
  "patient_id": "uuid",
  "patient_name": "John Doe",
  "patient_age": 45,
  "patient_gender": "Male",
  "blood_type": "A+",
  "status": "stable",
  "underlying_condition": {"hypertension": true, "diabetes": true},
  "phone_number": "+84912345678",
  "patient_created_at": "2025-01-01T10:00:00Z",
  
  "case_id": "uuid",
  "image_path": "https://bucket.s3.amazonaws.com/cases/patient-id.png",
  "processed_img_path": "https://bucket.s3.amazonaws.com/cases/patient-id/image.dicom",
  "case_timestamp": "2025-12-12T10:00:00Z",
  "diagnosis": "Pneumonia",
  "findings": "Infiltrates visible",
  
  "ai_result_id": "uuid",
  "predicted_diagnosis": "Pneumonia",
  "confident_score": null,
  "bounding_box": {
    "detections": [...]
  },
  "concepts": {
    "top_classes": [...],
    "detected_concepts": [...]
  },
  "ai_result_created_at": "2025-12-12T10:05:00Z",
  
  "report_id": "uuid",
  "model_report": {
    "MeSH": "...",
    "Findings": "...",
    "Impression": "..."
  },
  "doctor_report": null,
  "feedback_note": null,
  "report_created_at": "2025-12-12T10:10:00Z"
}
"""
