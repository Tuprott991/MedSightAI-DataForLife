"""
Test Report Generation API Endpoint
This script demonstrates how to use the /api/v1/reports/generate endpoint
"""
import requests
import json

# Configuration
BACKEND_URL = "http://localhost:8000"
CASE_ID = "0aedd4e9-fdc9-4c4c-bbce-e8d68e3478d1"  # Replace with actual case UUID

def test_generate_report():
    """
    Test the report generation endpoint
    
    Prerequisites:
    1. Case must exist in database
    2. CAM inference must be run first (ai_result must exist)
    3. Patient must have underlying_condition data
    """
    print("="*80)
    print("📄 Testing Report Generation API")
    print("="*80)
    
    # Endpoint URL
    url = f"{BACKEND_URL}/api/v1/reports/generate"
    
    # Request payload
    payload = {
        "case_id": CASE_ID
    }
    
    print(f"\n📡 Calling: POST {url}")
    print(f"📋 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Make request
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        print("\n✅ Success! Report Generated:")
        print(json.dumps(result, indent=2))
        
        # Display summary
        print("\n📊 Summary:")
        print(f"   Report ID: {result.get('id')}")
        print(f"   Case ID: {result.get('case_id')}")
        print(f"   Created At: {result.get('created_at')}")
        
        model_report = result.get('model_report', {})
        if model_report:
            print("\n📝 Report Details:")
            print(f"   MeSH: {model_report.get('MeSH')}")
            print(f"   Image: {model_report.get('Image')}")
            print(f"   Indication: {model_report.get('Indication')}")
            print(f"   Comparison: {model_report.get('Comparison')}")
            print(f"\n   Findings:")
            print(f"   {model_report.get('Findings')}")
            print(f"\n   Impression:")
            print(f"   {model_report.get('Impression')}")
        
        print("\n" + "="*80)
        
        return result
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    return None


def get_report(case_id):
    """
    Get existing report for a case
    """
    url = f"{BACKEND_URL}/api/v1/reports/{case_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        result = response.json()
        print(f"\n📄 Existing Report Found:")
        print(json.dumps(result, indent=2))
        return result
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"\n💡 No existing report found for case {case_id}")
        else:
            print(f"\n❌ Error: {e.response.text}")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def run_cam_inference_first(case_id):
    """
    Helper function to run CAM inference first if needed
    """
    url = f"{BACKEND_URL}/api/v1/analysis/cam-inference/{case_id}"
    
    print(f"\n🔬 Running CAM inference first for case {case_id}...")
    
    try:
        response = requests.post(url, params={"threshold": 0.5})
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ CAM inference completed!")
        print(f"   Detected concepts: {len(result.get('concepts', {}).get('detected_concepts', []))}")
        return result
        
    except Exception as e:
        print(f"❌ CAM inference failed: {e}")
        return None


if __name__ == "__main__":
    print("Step 1: Check for existing report...")
    existing_report = get_report(CASE_ID)
    
    if not existing_report:
        print("\nStep 2: Run CAM inference (prerequisite)...")
        ai_result = run_cam_inference_first(CASE_ID)
        
        if ai_result:
            print("\nStep 3: Generate report...")
            test_generate_report()
        else:
            print("\n⚠️  CAM inference failed. Cannot generate report.")
    else:
        print("\n💡 Report already exists. Generating new report will update it...")
        test_generate_report()


"""
Example Response:
{
  "id": "report-uuid",
  "case_id": "case-uuid",
  "model_report": {
    "MeSH": "Lung Opacity, Pulmonary fibrosis",
    "Image": "X-ray Chest PA",
    "Indication": "hypertension, diabetes",
    "Comparison": "None",
    "Findings": "The heart size is normal. The mediastinum and aorta are unremarkable. There is evidence of Lung Opacity in the lower lung zones. Findings demonstrate Pulmonary fibrosis in the upper lung zones.",
    "Impression": "There is evidence of Lung Opacity in the lower lung zones and Pulmonary fibrosis in the upper lung zones.",
    "raw_report_text": "MeSH: Lung Opacity, Pulmonary fibrosis\\nimage: X-ray Chest PA\\nindication: hypertension, diabetes\\ncomparison: None\\nfindings: The heart size is normal. The mediastinum and aorta are unremarkable. There is evidence of Lung Opacity in the lower lung zones. Findings demonstrate Pulmonary fibrosis in the upper lung zones.\\nimpression: There is evidence of Lung Opacity in the lower lung zones and Pulmonary fibrosis in the upper lung zones.\\n"
  },
  "doctor_report": null,
  "feedback_note": null,
  "created_at": "2025-12-12T12:00:00Z"
}
"""
