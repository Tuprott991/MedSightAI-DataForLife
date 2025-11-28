"""
S3 Path Utilities
Helper functions for constructing S3 paths following the standard folder structure
"""
from uuid import UUID
from typing import Optional
from app.config.settings import settings


class S3PathBuilder:
    """Builder class for constructing S3 paths"""
    
    # ==================== Patient Folders ====================
    
    @staticmethod
    def patient_profile_path(patient_id: UUID, filename: str) -> str:
        """
        Patient profile picture path
        Example: patients/{patient_id}/profile/avatar.jpg
        """
        return f"{settings.S3_PATIENTS_PREFIX}{patient_id}/profile/{filename}"
    
    # ==================== Case Folders ====================
    
    @staticmethod
    def case_original_image(case_id: UUID, filename: str) -> str:
        """
        Original medical image path
        Example: cases/{case_id}/original/xray_001.jpg
        """
        return f"{settings.S3_CASES_PREFIX}{case_id}/{settings.S3_ORIGINAL_IMAGES_PREFIX}{filename}"
    
    @staticmethod
    def case_processed_image(case_id: UUID, filename: str) -> str:
        """
        Processed/normalized image path
        Example: cases/{case_id}/processed/normalized_001.jpg
        """
        return f"{settings.S3_CASES_PREFIX}{case_id}/{settings.S3_PROCESSED_IMAGES_PREFIX}{filename}"
    
    @staticmethod
    def case_annotated_image(case_id: UUID, filename: str) -> str:
        """
        Annotated image (Grad-CAM, bounding boxes) path
        Example: cases/{case_id}/annotated/gradcam_001.jpg
        """
        return f"{settings.S3_CASES_PREFIX}{case_id}/{settings.S3_ANNOTATED_IMAGES_PREFIX}{filename}"
    
    @staticmethod
    def case_segmentation_mask(case_id: UUID, filename: str) -> str:
        """
        Segmentation mask path
        Example: cases/{case_id}/segmentation/mask_001.png
        """
        return f"{settings.S3_CASES_PREFIX}{case_id}/{settings.S3_SEGMENTATION_PREFIX}{filename}"
    
    @staticmethod
    def case_report(case_id: UUID, filename: str) -> str:
        """
        Case report (PDF) path
        Example: cases/{case_id}/reports/ai_report.pdf
        """
        return f"{settings.S3_CASES_PREFIX}{case_id}/{settings.S3_REPORTS_PREFIX}{filename}"
    
    @staticmethod
    def case_thumbnail(case_id: UUID, filename: str) -> str:
        """
        Case thumbnail path
        Example: cases/{case_id}/processed/thumbnail.jpg
        """
        return f"{settings.S3_CASES_PREFIX}{case_id}/{settings.S3_PROCESSED_IMAGES_PREFIX}thumbnail_{filename}"
    
    @staticmethod
    def case_metadata(case_id: UUID) -> str:
        """
        Case metadata JSON path
        Example: cases/{case_id}/original/metadata.json
        """
        return f"{settings.S3_CASES_PREFIX}{case_id}/{settings.S3_ORIGINAL_IMAGES_PREFIX}metadata.json"
    
    # ==================== Education Mode Folders ====================
    
    @staticmethod
    def education_student_upload(session_id: UUID, filename: str) -> str:
        """
        Student uploaded image path
        Example: education/{session_id}/student_uploads/practice_xray.jpg
        """
        return f"{settings.S3_EDUCATION_PREFIX}{session_id}/{settings.S3_STUDENT_UPLOADS_PREFIX}{filename}"
    
    @staticmethod
    def education_student_annotation(session_id: UUID, filename: str) -> str:
        """
        Student annotation path
        Example: education/{session_id}/student_annotations/annotated_by_student.jpg
        """
        return f"{settings.S3_EDUCATION_PREFIX}{session_id}/{settings.S3_STUDENT_ANNOTATIONS_PREFIX}{filename}"
    
    @staticmethod
    def education_feedback(session_id: UUID, filename: str) -> str:
        """
        AI feedback image path
        Example: education/{session_id}/feedback/corrected_annotation.jpg
        """
        return f"{settings.S3_EDUCATION_PREFIX}{session_id}/{settings.S3_FEEDBACK_PREFIX}{filename}"
    
    # ==================== Similar Cases Folders ====================
    
    @staticmethod
    def similar_case_thumbnail(case_id: UUID, similar_case_id: UUID) -> str:
        """
        Similar case thumbnail path
        Example: similar_cases/{case_id}/thumbnails/similar_{similar_case_id}_thumb.jpg
        """
        return f"{settings.S3_SIMILAR_CASES_PREFIX}{case_id}/{settings.S3_THUMBNAILS_PREFIX}similar_{similar_case_id}_thumb.jpg"
    
    # ==================== Temporary & Exports ====================
    
    @staticmethod
    def temp_upload(filename: str) -> str:
        """
        Temporary upload path (for processing before moving to final location)
        Example: temp/uploads/{timestamp}_{uuid}.jpg
        Note: Set up S3 lifecycle policy to delete after 30 days
        """
        return f"{settings.S3_TEMP_PREFIX}{filename}"
    
    @staticmethod
    def export_file(export_id: UUID, filename: str) -> str:
        """
        Export file path (bulk reports, datasets)
        Example: exports/{export_id}/batch_report.pdf
        """
        return f"{settings.S3_EXPORTS_PREFIX}{export_id}/{filename}"
    
    # ==================== Bulk Operations ====================
    
    @staticmethod
    def list_case_images(case_id: UUID, image_type: str = "original") -> str:
        """
        Get prefix for listing all images of a specific type for a case
        
        Args:
            case_id: Case UUID
            image_type: 'original', 'processed', 'annotated', or 'segmentation'
        
        Returns:
            S3 prefix for listing
        """
        type_mapping = {
            "original": settings.S3_ORIGINAL_IMAGES_PREFIX,
            "processed": settings.S3_PROCESSED_IMAGES_PREFIX,
            "annotated": settings.S3_ANNOTATED_IMAGES_PREFIX,
            "segmentation": settings.S3_SEGMENTATION_PREFIX
        }
        
        prefix = type_mapping.get(image_type, settings.S3_ORIGINAL_IMAGES_PREFIX)
        return f"{settings.S3_CASES_PREFIX}{case_id}/{prefix}"
    
    @staticmethod
    def case_folder(case_id: UUID) -> str:
        """
        Get base folder path for a case (for listing all files)
        Example: cases/{case_id}/
        """
        return f"{settings.S3_CASES_PREFIX}{case_id}/"
    
    @staticmethod
    def education_session_folder(session_id: UUID) -> str:
        """
        Get base folder path for education session
        Example: education/{session_id}/
        """
        return f"{settings.S3_EDUCATION_PREFIX}{session_id}/"


# ==================== Convenience Functions ====================

def get_case_images_structure(case_id: UUID) -> dict:
    """
    Get all S3 paths for a case in a structured format
    Useful for frontend to know where to fetch images
    """
    return {
        "original": S3PathBuilder.list_case_images(case_id, "original"),
        "processed": S3PathBuilder.list_case_images(case_id, "processed"),
        "annotated": S3PathBuilder.list_case_images(case_id, "annotated"),
        "segmentation": S3PathBuilder.list_case_images(case_id, "segmentation"),
        "reports": f"{settings.S3_CASES_PREFIX}{case_id}/{settings.S3_REPORTS_PREFIX}",
        "metadata": S3PathBuilder.case_metadata(case_id)
    }


def parse_s3_path(s3_path: str) -> dict:
    """
    Parse an S3 path to extract components
    
    Example:
        Input: "cases/123e4567-e89b-12d3-a456-426614174000/original/xray_001.jpg"
        Output: {
            "type": "case",
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "category": "original",
            "filename": "xray_001.jpg"
        }
    """
    parts = s3_path.split("/")
    
    if len(parts) < 2:
        return {"type": "unknown", "path": s3_path}
    
    folder_type = parts[0]
    
    if folder_type == "cases":
        return {
            "type": "case",
            "case_id": parts[1] if len(parts) > 1 else None,
            "category": parts[2] if len(parts) > 2 else None,
            "filename": parts[3] if len(parts) > 3 else None,
            "full_path": s3_path
        }
    
    elif folder_type == "education":
        return {
            "type": "education",
            "session_id": parts[1] if len(parts) > 1 else None,
            "category": parts[2] if len(parts) > 2 else None,
            "filename": parts[3] if len(parts) > 3 else None,
            "full_path": s3_path
        }
    
    elif folder_type == "patients":
        return {
            "type": "patient",
            "patient_id": parts[1] if len(parts) > 1 else None,
            "category": parts[2] if len(parts) > 2 else None,
            "filename": parts[3] if len(parts) > 3 else None,
            "full_path": s3_path
        }
    
    else:
        return {
            "type": folder_type,
            "path": s3_path
        }


# Create singleton instance for easy import
s3_paths = S3PathBuilder()
