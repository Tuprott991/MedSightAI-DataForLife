"""
Test CAM Inference API Endpoint
This script demonstrates how to use the new /api/v1/analysis/cam-inference/{case_id} endpoint
"""
import requests
import json

# Configuration
BACKEND_URL = "http://localhost:8000"
CASE_ID = "243006e4-9480-4a95-89b4-86ac32dd03c6"  # Replace with actual case UUID
THRESHOLD = 0.5

def test_cam_inference():
    """
    Test the CAM inference endpoint
    """
    print("="*80)
    print("🧪 Testing CAM Inference API")
    print("="*80)
    
    # Endpoint URL
    url = f"{BACKEND_URL}/api/v1/analysis/cam-inference/{CASE_ID}"
    
    # Parameters
    params = {
        "threshold": THRESHOLD
    }
    
    print(f"\n📡 Calling: POST {url}")
    print(f"📋 Parameters: {params}")
    
    try:
        # Make request
        response = requests.post(url, params=params)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        print("\n✅ Success! AI Result:")
        print(json.dumps(result, indent=2))
        
        # Display summary
        print("\n📊 Summary:")
        print(f"   Case ID: {result.get('case_id')}")
        print(f"   Diagnosis: {result.get('predicted_diagnosis')}")
        
        concepts_data = result.get('concepts', {})
        detected_concepts = concepts_data.get('detected_concepts', [])
        print(f"   Concepts detected: {len(detected_concepts)}")
        
        if detected_concepts:
            print("\n🔍 Detected Abnormalities:")
            detections = result.get('bounding_box', {}).get('detections', [])
            for detection in detections:
                concept = detection.get('concept')
                bbox = detection.get('bbox')
                prob = detection.get('probability')
                class_idx = detection.get('class_idx')
                print(f"   • {concept}")
                print(f"     Class: {class_idx} | Probability: {prob:.4f}")
                if bbox:
                    print(f"     BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        
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


def get_all_cases():
    """
    Helper function to get list of cases
    """
    url = f"{BACKEND_URL}/api/v1/cases/"
    params = {"page": 1, "page_size": 10}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print("📋 Available Cases:")
        for case in data.get('cases', []):
            print(f"   - {case['id']} (Patient: {case['patient_id']})")
        
        return data.get('cases', [])
        
    except Exception as e:
        print(f"❌ Error fetching cases: {e}")
        return []


if __name__ == "__main__":
    # First, get available cases
    print("Fetching available cases...\n")
    cases = get_all_cases()
    
    if cases:
        print(f"\n💡 Update CASE_ID variable with one of the above case IDs")
        print(f"   Example: CASE_ID = '{cases[0]['id']}'")
    
    # If CASE_ID is set, run the test
    if CASE_ID != "YOUR_CASE_ID_HERE":
        test_cam_inference()
    else:
        print("\n⚠️  Please set CASE_ID variable in the script first!")


"""
Example Response:
{
  "id": "ai-result-uuid",
  "case_id": "case-uuid",
  "predicted_diagnosis": "Pneumonia",
  "confident_score": null,
  "bounding_box": {
    "detections": [
      {
        "bbox": [65, 74, 303, 187],
        "concept": "Cardiomegaly",
        "class_idx": 3,
        "probability": 0.9998501539230347
      },
      {
        "bbox": [47, 111, 112, 180],
        "concept": "Aortic enlargement",
        "class_idx": 0,
        "probability": 0.9996201992034912
      }
    ]
  },
  "concepts": {
    "top_classes": [
      {
        "prob": 0.9998501539230347,
        "concepts": "Cardiomegaly",
        "class_idx": 3
      },
      {
        "prob": 0.9996201992034912,
        "concepts": "Aortic enlargement",
        "class_idx": 0
      }
    ],
    "detected_concepts": ["Cardiomegaly", "Aortic enlargement"]
  },
  "created_at": "2025-12-12T10:00:00Z"
}
"""
