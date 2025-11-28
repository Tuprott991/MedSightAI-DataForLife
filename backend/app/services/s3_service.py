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


class S3Service:
    """Service for S3 operations"""
    
    def __init__(self):
        self.client = get_s3_client()
        self.bucket_name = settings.S3_BUCKET_NAME
    
    def upload_file(
        self,
        file: UploadFile,
        prefix: str = "",
        custom_name: Optional[str] = None
    ) -> str:
        """
        Upload a file to S3
        
        Args:
            file: UploadFile object from FastAPI
            prefix: S3 prefix/folder (e.g., 'original/', 'processed/')
            custom_name: Custom filename (optional)
        
        Returns:
            S3 file path
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
            
            # Upload to S3
            self.client.upload_fileobj(
                file.file,
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': file.content_type or 'application/octet-stream'}
            )
            
            return s3_key
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {str(e)}")
    
    def upload_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        prefix: str = "",
        content_type: str = "image/jpeg"
    ) -> str:
        """
        Upload bytes data to S3
        
        Args:
            file_bytes: Binary data
            filename: Filename
            prefix: S3 prefix/folder
            content_type: Content type
        
        Returns:
            S3 file path
        """
        try:
            s3_key = f"{prefix}{filename}"
            
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_bytes,
                ContentType=content_type
            )
            
            return s3_key
        
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
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for temporary file access
        
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
