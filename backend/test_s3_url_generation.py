"""
Direct test of S3 service to verify URL generation
"""
import sys
sys.path.append('.')

from app.services.s3_service import s3_service
from app.config.settings import settings

print("Testing S3 Service URL Generation\n")
print("="*60)

# Test the get_public_url method directly
test_key = "cases/test-patient-id/original/test-image.jpg"

print(f"Bucket: {settings.S3_BUCKET_NAME}")
print(f"Region: {settings.AWS_REGION}")
print(f"Test S3 Key: {test_key}")
print()

public_url = s3_service.get_public_url(test_key)

print(f"Generated Public URL:")
print(f"{public_url}")
print()

expected_url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{test_key}"
print(f"Expected URL:")
print(f"{expected_url}")
print()

if public_url == expected_url:
    print("✅ URL generation is working correctly!")
else:
    print("❌ URL mismatch!")

print("\n" + "="*60)
print("If this shows the correct URL, restart the server and test again.")
