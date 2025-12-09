-- Migration: Add new columns to patient and cases tables
-- Date: 2025-12-09
-- Description: Adds blood_type, status, underlying_condition to patient table
--              Adds diagnosis, findings to cases table

-- ============================================
-- Add columns to patient table
-- ============================================

-- Add blood_type column (Text)
ALTER TABLE patient 
ADD COLUMN IF NOT EXISTS blood_type TEXT;

-- Add status column (Text with enum-like values: stable, improving, critical)
ALTER TABLE patient 
ADD COLUMN IF NOT EXISTS status TEXT;

-- Add underlying_condition column (JSONB for structured data)
ALTER TABLE patient 
ADD COLUMN IF NOT EXISTS underlying_condition JSONB;

-- ============================================
-- Add columns to cases table
-- ============================================

-- Add diagnosis column (Text for disease diagnosis)
ALTER TABLE cases 
ADD COLUMN IF NOT EXISTS diagnosis TEXT;

-- Add findings column (Text for extended notes)
ALTER TABLE cases 
ADD COLUMN IF NOT EXISTS findings TEXT;

-- ============================================
-- Add comments for documentation
-- ============================================

COMMENT ON COLUMN patient.blood_type IS 'Blood type (A+, B-, O+, AB+, etc.)';
COMMENT ON COLUMN patient.status IS 'Patient status: stable, improving, or critical';
COMMENT ON COLUMN patient.underlying_condition IS 'Chronic or underlying medical conditions as JSON';
COMMENT ON COLUMN patient.history IS 'Medical history with dates: {"MM-DD-YYYY": {"diagnosis": "...", "findings": "..."}}';

COMMENT ON COLUMN cases.diagnosis IS 'Disease diagnosis (e.g., Pneumonia, TB, COVID-19)';
COMMENT ON COLUMN cases.findings IS 'Extended clinical notes and findings';

-- ============================================
-- Verification queries
-- ============================================

-- Check patient table columns
-- SELECT column_name, data_type, is_nullable 
-- FROM information_schema.columns 
-- WHERE table_name = 'patient';

-- Check cases table columns
-- SELECT column_name, data_type, is_nullable 
-- FROM information_schema.columns 
-- WHERE table_name = 'cases';
