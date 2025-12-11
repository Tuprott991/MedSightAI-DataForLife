"""
S3 Storage Service
Handles file upload, download, and deletion from AWS S3
"""
import os
import io
from typing import Optional, BinaryIO, List
from datetime import datetime
from uuid import uuid4, UUID
from fastapi import UploadFile, HTTPException
from app.config.s3 import get_s3_client
from app.config.settings import settings
from app.utils.s3_paths import S3PathBuilder
import pydicom
from PIL import Image
import numpy as np


class S3Service:
    """Service for S3 operations"""
    
    def __init__(self):
        self.client = get_s3_client()
        self.bucket_name = settings.S3_BUCKET_NAME
    
    def dicom_to_png_bytes(self, dicom_bytes: bytes) -> bytes:
        """
        Convert DICOM bytes to PNG bytes
        
        Args:
            dicom_bytes: Raw DICOM file bytes
        
        Returns:
            PNG image bytes
        """
        try:
            # Load DICOM from bytes
            dicom_file = pydicom.dcmread(io.BytesIO(dicom_bytes))
            
            # Get pixel array
            pixel_array = dicom_file.pixel_array
            
            # Normalize to 0-255 range
            pixel_array = pixel_array.astype(float)
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0
            pixel_array = pixel_array.astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(pixel_array)
            
            # Convert to RGB if grayscale
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            return buffer.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to convert DICOM to PNG: {str(e)}")
    
    def upload_dicom_and_convert(
        self,
        file: UploadFile,
        patient_id: str
    ) -> tuple[str, str]:
        """
        Upload DICOM file and automatically convert to PNG
        Stores both files with standardized naming:
        - DICOM: s3://bucket/cases/{patient_id}/image.dicom
        - PNG: s3://bucket/cases/{patient_id}.png
        
        Args:
            file: UploadFile object (DICOM file)
            patient_id: Patient UUID as string
        
        Returns:
            Tuple of (dicom_url, png_url)
        """
        try:
            # Read file bytes
            file_bytes = file.file.read()
            file.file.seek(0)  # Reset file pointer
            
            # Upload DICOM file
            dicom_key = f"cases/{patient_id}/image.dicom"
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=dicom_key,
                Body=file_bytes,
                ContentType="application/dicom"
            )
            dicom_url = self.get_public_url(dicom_key)
            
            # Convert DICOM to PNG
            png_bytes = self.dicom_to_png_bytes(file_bytes)
            
            # Upload PNG file
            png_key = f"cases/{patient_id}.png"
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=png_key,
                Body=png_bytes,
                ContentType="image/png"
            )
            png_url = self.get_public_url(png_key)
            
            return dicom_url, png_url
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload and convert DICOM: {str(e)}")
    
    def upload_file(
        self,
        file: UploadFile,
        prefix: str = "",
        custom_name: Optional[str] = None,
        make_public: bool = True
    ) -> str:
        """
        Upload a file to S3
        
        Args:
            file: UploadFile object from FastAPI
            prefix: S3 prefix/folder (e.g., 'original/', 'processed/')
            custom_name: Custom filename (optional)
            make_public: Make file publicly accessible (default: True)
        
        Returns:
            Full S3 public URL (e.g., https://bucket.s3.region.amazonaws.com/key)
        """
        try:
            # Generate unique filename
            if custom_name:
                filename = custom_name
            else:
                ext = os.path.splitext(file.filename)[1]
                filename = f"{uuid4()}{ext}"
            
            # Construct S3 key
            s3_key = f"{prefix}{filename}"
            
            # Prepare extra args
            extra_args = {
                'ContentType': file.content_type or 'application/octet-stream'
            }
            
            # Note: Not using ACL - bucket policy will handle public access
            # If ACL is needed, enable ACLs in S3 bucket settings first
            
            # Upload to S3
            self.client.upload_fileobj(
                file.file,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            # Return full public URL
            return self.get_public_url(s3_key)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {str(e)}")
    
    def upload_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        prefix: str = "",
        content_type: str = "image/jpeg",
        make_public: bool = True
    ) -> str:
        """
        Upload bytes data to S3
        
        Args:
            file_bytes: Binary data
            filename: Filename
            prefix: S3 prefix/folder
            content_type: Content type
            make_public: Make file publicly accessible (default: True)
        
        Returns:
            Full S3 public URL
        """
        try:
            s3_key = f"{prefix}{filename}"
            
            put_args = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'Body': file_bytes,
                'ContentType': content_type
            }
            
            # Note: Not using ACL - bucket policy will handle public access
            # If ACL is needed, enable ACLs in S3 bucket settings first
            
            self.client.put_object(**put_args)
            
            # Return full public URL
            return self.get_public_url(s3_key)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload bytes to S3: {str(e)}")
    
    def download_file(self, s3_key: str) -> bytes:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            File bytes
        """
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
        
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"File not found in S3: {str(e)}")
    
    def download_to_stream(self, s3_key: str) -> io.BytesIO:
        """
        Download file to BytesIO stream
        
        Args:
            s3_key: S3 object key
        
        Returns:
            BytesIO stream
        """
        try:
            stream = io.BytesIO()
            self.client.download_fileobj(self.bucket_name, s3_key, stream)
            stream.seek(0)
            return stream
        
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"File not found in S3: {str(e)}")
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file from S3: {str(e)}")
    
    def get_public_url(self, s3_key: str) -> str:
        """
        Get the public URL for an S3 object (no expiration)
        
        Args:
            s3_key: S3 object key (e.g., 'cases/patient-id/original/image.jpg')
        
        Returns:
            Public URL (e.g., https://bucket.s3.region.amazonaws.com/key)
        """
        # Get region from settings
        region = settings.AWS_REGION
        
        # Construct public URL
        if region == 'us-east-1':
            # us-east-1 uses different URL format
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
        else:
            url = f"https://{self.bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
        
        return url
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for temporary file access
        (Use get_public_url() instead for permanent access)
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
        
        Returns:
            Presigned URL
        """
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}")
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            True if file exists
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False
    
    def get_file_size(self, s3_key: str) -> int:
        """
        Get file size from S3
        
        Args:
            s3_key: S3 object key
        
        Returns:
            File size in bytes
        """
        try:
            response = self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response['ContentLength']
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    
    def list_files(self, prefix: str) -> List[dict]:
        """
        List all files under a prefix
        
        Args:
            prefix: S3 prefix to list
        
        Returns:
            List of file objects with key, size, last_modified
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'url': self.get_presigned_url(obj['Key'])
                })
            
            return files
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")
    
    def copy_file(self, source_key: str, destination_key: str) -> bool:
        """
        Copy a file within S3
        
        Args:
            source_key: Source S3 key
            destination_key: Destination S3 key
        
        Returns:
            True if successful
        """
        try:
            copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
            self.client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=destination_key
            )
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to copy file: {str(e)}")
    
    def move_file(self, source_key: str, destination_key: str) -> bool:
        """
        Move a file within S3 (copy then delete)
        
        Args:
            source_key: Source S3 key
            destination_key: Destination S3 key
        
        Returns:
            True if successful
        """
        try:
            self.copy_file(source_key, destination_key)
            self.delete_file(source_key)
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to move file: {str(e)}")
    
    # ==================== High-Level Case Operations ====================
    
    def upload_case_original_image(
        self,
        case_id: UUID,
        file: UploadFile,
        custom_name: Optional[str] = None
    ) -> str:
        """Upload original medical image for a case"""
        filename = custom_name or f"{uuid4()}{os.path.splitext(file.filename)[1]}"
        s3_key = S3PathBuilder.case_original_image(case_id, filename)
        
        self.client.upload_fileobj(
            file.file,
            self.bucket_name,
            s3_key,
            ExtraArgs={'ContentType': file.content_type or 'image/jpeg'}
        )
        return s3_key
    
    def upload_case_processed_image(
        self,
        case_id: UUID,
        image_bytes: bytes,
        filename: str,
        content_type: str = "image/jpeg"
    ) -> str:
        """Upload processed/normalized image"""
        s3_key = S3PathBuilder.case_processed_image(case_id, filename)
        return self.upload_bytes(image_bytes, "", prefix=s3_key, content_type=content_type)
    
    def upload_case_annotated_image(
        self,
        case_id: UUID,
        image_bytes: bytes,
        filename: str,
        content_type: str = "image/jpeg"
    ) -> str:
        """Upload annotated image (Grad-CAM, bounding boxes)"""
        s3_key = S3PathBuilder.case_annotated_image(case_id, filename)
        return self.upload_bytes(image_bytes, "", prefix=s3_key, content_type=content_type)
    
    def upload_case_report(
        self,
        case_id: UUID,
        report_bytes: bytes,
        filename: str
    ) -> str:
        """Upload case report PDF"""
        s3_key = S3PathBuilder.case_report(case_id, filename)
        return self.upload_bytes(report_bytes, "", prefix=s3_key, content_type="application/pdf")
    
    def get_case_images(self, case_id: UUID, image_type: str = "original") -> List[dict]:
        """Get all images of a specific type for a case"""
        prefix = S3PathBuilder.list_case_images(case_id, image_type)
        return self.list_files(prefix)
    
    def delete_case_folder(self, case_id: UUID) -> bool:
        """Delete all files for a case"""
        try:
            prefix = S3PathBuilder.case_folder(case_id)
            files = self.list_files(prefix)
            
            for file in files:
                self.delete_file(file['key'])
            
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete case folder: {str(e)}")
    
    # ==================== Education Mode Operations ====================
    
    def upload_student_image(
        self,
        session_id: UUID,
        file: UploadFile
    ) -> str:
        """Upload student practice image"""
        filename = f"{uuid4()}{os.path.splitext(file.filename)[1]}"
        s3_key = S3PathBuilder.education_student_upload(session_id, filename)
        
        self.client.upload_fileobj(
            file.file,
            self.bucket_name,
            s3_key,
            ExtraArgs={'ContentType': file.content_type or 'image/jpeg'}
        )
        return s3_key
    
    def upload_student_annotation(
        self,
        session_id: UUID,
        image_bytes: bytes,
        filename: str
    ) -> str:
        """Upload student's annotation"""
        s3_key = S3PathBuilder.education_student_annotation(session_id, filename)
        return self.upload_bytes(image_bytes, "", prefix=s3_key, content_type="image/jpeg")
    
    def upload_feedback_image(
        self,
        session_id: UUID,
        image_bytes: bytes,
        filename: str
    ) -> str:
        """Upload AI feedback image"""
        s3_key = S3PathBuilder.education_feedback(session_id, filename)
        return self.upload_bytes(image_bytes, "", prefix=s3_key, content_type="image/jpeg")


# Create singleton instance
s3_service = S3Service()
