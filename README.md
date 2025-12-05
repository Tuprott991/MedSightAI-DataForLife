# MedSight AI - Explainable X-Ray Diagnosis Platform

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)
![React](https://img.shields.io/badge/React-18.2.0-61DAFB.svg)

**AI-powered chest X-ray diagnosis with explainable AI (xAI) for doctors and medical students**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [API Documentation](#-api-documentation) • [Team](#-team)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Overview

**MedSight AI** is an advanced medical imaging platform that applies **Explainable Artificial Intelligence (xAI)** to assist doctors in diagnosing chest X-ray images and provide an interactive learning environment for medical students.

### Product Information

- **Product Name**: AI-XRay Explainable Diagnosis Application
- **Field**: AI in Healthcare / Medical Imaging / Explainable AI
- **Development Team**: SoftAI
- **Version**: 1.0.0

### Key Highlights

- 🔍 **Explainable AI**: Transparent diagnosis with visual explanations using Grad-CAM and concept-based reasoning
- 🏥 **Doctor Support Mode**: AI-assisted diagnosis with similar case retrieval and automated report generation
- 🎓 **Education Mode**: Interactive learning platform for medical students with scoring and feedback
- 🌐 **Multi-modal Analysis**: Combines image analysis with clinical data for comprehensive diagnosis
- 📊 **Knowledge Graph Integration**: Automated medical report generation using MedGemma LLM

---

## 🔴 Problem Statement

In the healthcare sector's digital transformation, AI in medical imaging is becoming increasingly common. However, most current AI models operate as **"black boxes"** - providing predictions without clear explanations.

### Challenges

- ❌ **Lack of Transparency**: Models don't explain their decision-making process
- ❌ **Limited Trust**: Doctors cannot verify AI reasoning
- ❌ **Difficult Interpretation**: No clear indication of lesion locations and severity
- ❌ **Training Gap**: Students lack interactive tools to learn AI-assisted diagnosis

### Impact on Healthcare

Chest X-rays are crucial for screening and early detection of:
- Pneumonia
- Pulmonary Tuberculosis
- Fibrosis
- COVID-19
- Pleural Effusion

Without explainable AI, doctors struggle to assess causes, severity, and lesion locations, limiting AI adoption in clinical practice.

---

## ✨ Features

### 🏥 Diagnosis Support Mode (For Doctors)

#### 1. **Clinical Input & Image Upload**
- Enter patient symptoms (cough, fever, chest pain, etc.)
- Record medical history (TB, lung disease exposure, etc.)
- Upload chest X-ray images (JPEG, PNG, DICOM)
- Input relevant test results (Mantoux, GeneXpert, CRP, WBC)

#### 2. **AI-Powered Analysis**
- **Automatic Preprocessing**: Image normalization, noise removal, contrast enhancement
- **Disease Detection**: CNN/ViT models identify abnormal regions
- **Lesion Classification**: Nodules, infiltration, pleural effusion, reticular patterns, etc.
- **Probability Scores**: Confidence levels for each detected abnormality

#### 3. **Explainable AI (xAI) Layer**
- **Grad-CAM Heatmaps**: Visual highlighting of suspected lesion areas
- **Concept-Based Explanation**: Maps features to medical concepts:
  - Consolidation
  - Cavity formation
  - Fibrosis
  - Opacity patterns
  - Tissue damage
- **Prototype Comparison**: Matches detected patterns with known pathological concepts

#### 4. **Similar Case Retrieval**
- **Vector-Based Search**: Uses MedSigLip embeddings (1152 dimensions)
- **Top-K Similar Cases**: Retrieves most similar diagnosed cases from database
- **Visual Comparison**: Side-by-side view with similarity scores
- **Reference Learning**: Helps doctors compare current case with known cases

#### 5. **AI-Assisted Medical Reasoning**
- **Automated Report Generation**: MedGemma LLM + Knowledge Graph
- **Preliminary Diagnosis**: Suggested diagnoses with reasoning
- **Morphological Description**: Detailed lesion characteristics
- **AI Reasoning Summary**: Explanation of diagnostic logic

#### 6. **Human-in-the-Loop Learning**
- **Doctor Feedback**: Adjust detected areas and add clinical notes
- **Expert Corrections**: Update conclusions after test results
- **Model Improvement**: System learns from expert feedback

### 🎓 Education Mode (For Students)

#### 1. **Practice Sessions**
- Select disease type to practice (TB, pneumonia, effusion, etc.)
- Receive unlabeled X-ray images
- Identify suspected lesion areas
- Record preliminary diagnosis

#### 2. **Interactive Assessment**
- **BBox Matching**: Compare student's identified areas with ground truth
- **Diagnosis Accuracy**: Evaluate correctness of disease classification
- **Automated Scoring**: Real-time grading based on accuracy

#### 3. **Learning Feedback**
- **Visual Heatmap**: Show correct lesion locations
- **Explanation**: Why the answer is correct/incorrect
- **Concept Learning**: Understand diagnostic reasoning
- **Performance Tracking**: Monitor progress over time

---

## 🏗 Architecture

### System Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE (React)                      │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │  Doctor Mode     │              │  Education Mode  │         │
│  │  - Upload Image  │              │  - Practice      │         │
│  │  - View Results  │              │  - Submit Answer │         │
│  │  - Feedback      │              │  - Get Scores    │         │
│  └──────────────────┘              └──────────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼────────────────────────────────────┐
│                   BACKEND (FastAPI)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  API Layer   │  │   Services   │  │   Database   │          │
│  │  - Cases     │  │  - AI Model  │  │  - PostgreSQL│          │
│  │  - Analysis  │  │  - S3 Storage│  │  - Zilliz    │          │
│  │  - Education │  │  - Embeddings│  │  - Vector DB │          │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘          │
└─────────┼──────────────────┼─────────────────────────────────────┘
          │                  │
┌─────────▼──────────────────▼─────────────────────────────────────┐
│                    AI PROCESSING LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Image       │  │  Explainable │  │  Medical LLM │          │
│  │  Preprocessing│  │  AI (xAI)   │  │  (MedGemma)  │          │
│  │              │  │  - Grad-CAM  │  │  + Knowledge │          │
│  │  - Normalize │  │  - Concepts  │  │    Graph     │          │
│  │  - Denoise   │  │  - Prototype │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                   │
│  ┌──────▼──────────────────▼──────────────────▼───────┐          │
│  │         CNN/ViT Backbone (DenseNet121)             │          │
│  │         - Feature Extraction                       │          │
│  │         - Lesion Detection & Classification        │          │
│  └────────────────────────────────────────────────────┘          │
└───────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM COMPONENTS                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Image Preprocessing Module                               │
│    - Normalization, brightness adjustment, noise filtering  │
├─────────────────────────────────────────────────────────────┤
│ 2. AI Diagnosis Module                                      │
│    - Abnormality detection and classification               │
│    - Backbone: DenseNet121 / ResNet50 / ViT                │
├─────────────────────────────────────────────────────────────┤
│ 3. xAI Module (Explainability Layer)                        │
│    - Grad-CAM heatmap generation                            │
│    - Concept Bottleneck Model                               │
│    - Prototype Learning                                     │
├─────────────────────────────────────────────────────────────┤
│ 4. Med-Gemma / LLM Module                                   │
│    - Medical report generation                              │
│    - Knowledge Graph integration                            │
│    - Reasoning explanation                                  │
├─────────────────────────────────────────────────────────────┤
│ 5. User Interface (UI)                                      │
│    - Image upload and viewing                               │
│    - Result visualization                                   │
│    - Feedback collection                                    │
├─────────────────────────────────────────────────────────────┤
│ 6. Education Module                                         │
│    - Practice case selection                                │
│    - Student answer scoring                                 │
│    - Performance evaluation and feedback                    │
├─────────────────────────────────────────────────────────────┤
│ 7. Database Layer                                           │
│    - PostgreSQL: Patient data, cases, AI results           │
│    - Zilliz Cloud: Vector embeddings for similarity search │
│    - AWS S3: Medical image storage                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠 Technology Stack

### Backend
- **Framework**: FastAPI 0.109.0
- **Language**: Python 3.11+
- **Database**: PostgreSQL (Neon Cloud)
- **Vector Database**: Zilliz Cloud (Serverless Milvus)
- **Storage**: AWS S3
- **ORM**: SQLAlchemy 2.0.25
- **Validation**: Pydantic 2.5.3
- **API Server**: Uvicorn

### AI/ML Models
- **Backbone Models**: 
  - DenseNet121
  - ResNet50
  - Vision Transformer (ViT)
- **Explainability**: 
  - Grad-CAM
  - Concept Bottleneck Model
  - Prototype Learning
- **Embeddings**: MedSigLip (1152 dimensions)
- **LLM**: MedGemma + Knowledge Graph
- **Frameworks**: PyTorch, TensorFlow

### Frontend
- **Framework**: React 18.2.0
- **Desktop**: Electron (Cross-platform)
- **Build Tool**: Vite 5.4.0
- **UI Library**: Custom components with Tailwind CSS

### Infrastructure
- **Cloud Provider**: AWS (S3), Neon (PostgreSQL), Zilliz (Vector DB)
- **API Documentation**: OpenAPI/Swagger
- **Container**: Docker support (optional)

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Node.js 16+ and npm
- PostgreSQL (Neon Cloud account)
- AWS Account (for S3)
- Zilliz Cloud account

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/Tuprott991/SoftAI---DataForLife---MedSightAI.git
cd SoftAI---DataForLife---MedSightAI
```

2. **Set up Python virtual environment**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials:
# - DATABASE_URL (Neon PostgreSQL)
# - AWS credentials (S3)
# - ZILLIZ_CLOUD_URI and API_KEY
```

5. **Initialize database**
```bash
python -c "from app.config.database import init_db; init_db()"
```

6. **Test connections**
```bash
python test_db_connection.py
python test_s3_connection.py
python test_zilliz_connection.py
```

7. **Start the backend server**
```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/v1/health`

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd ../frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start development server**
```bash
npm run start
```

The Electron app will launch automatically.

---

## 🚀 Usage

### Doctor Mode

1. **Create a Patient**
   - Navigate to "Patients" section
   - Click "Add New Patient"
   - Enter patient information (name, age, gender, medical history)

2. **Upload X-Ray Image**
   - Select patient
   - Click "New Case"
   - Upload chest X-ray image
   - Add clinical notes and symptoms

3. **Run AI Analysis**
   - Click "Analyze" on the case
   - System performs:
     - Image preprocessing
     - AI inference
     - Grad-CAM generation
     - Similar case retrieval
     - Report generation

4. **Review Results**
   - View detected lesions with heatmap overlay
   - Check AI confidence scores
   - Browse similar cases
   - Read AI-generated preliminary report

5. **Provide Feedback**
   - Adjust detected areas if needed
   - Add diagnostic notes
   - Confirm or update diagnosis

### Education Mode

1. **Select Practice Type**
   - Choose disease category (TB, pneumonia, etc.)
   - Select difficulty level

2. **Complete Practice Case**
   - View unlabeled X-ray image
   - Draw bounding boxes around suspected lesions
   - Submit diagnosis

3. **Receive Feedback**
   - View your score
   - See correct lesion locations
   - Read explanation of the diagnosis
   - Track your progress

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Key Endpoints

#### Patient Management
- `POST /patients/` - Create new patient
- `GET /patients/` - List all patients
- `GET /patients/{patient_id}` - Get patient details
- `PUT /patients/{patient_id}` - Update patient
- `DELETE /patients/{patient_id}` - Delete patient

#### Case Management
- `POST /cases/` - Create new case (JSON with image_path)
- `POST /cases/upload` - Upload case with file
- `GET /cases/` - List cases
- `GET /cases/{case_id}` - Get case details
- `GET /cases/{case_id}/image-url` - Get presigned image URL

#### AI Analysis
- `POST /analysis/full-pipeline` - Run complete AI analysis
- `POST /analysis/preprocess` - Preprocess image only
- `POST /analysis/inference` - Run AI inference
- `POST /analysis/gradcam` - Generate Grad-CAM heatmap

#### Similar Cases
- `POST /similarity/search` - Find similar cases
- `POST /similarity/embed` - Generate embeddings

#### Education
- `GET /education/practice-cases` - Get practice cases
- `POST /education/submit-answer` - Submit student answer
- `POST /education/chat` - Chat with AI tutor

#### Health Check
- `GET /health` - Check system health
- `GET /version` - Get API version

### Example Request

**Create Case (JSON)**
```bash
curl -X POST "http://localhost:8000/api/v1/cases/" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "550e8400-e29b-41d4-a716-446655440000",
    "image_path": "cases/patient-id/original/xray_001.jpg",
    "processed_img_path": null,
    "similar_cases": [],
    "similarity_scores": []
  }'
```

**Response**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "patient_id": "550e8400-e29b-41d4-a716-446655440000",
  "image_path": "cases/patient-id/original/xray_001.jpg",
  "processed_img_path": null,
  "timestamp": "2025-12-05T10:30:00Z",
  "similar_cases": [],
  "similarity_scores": []
}
```

Full API documentation: `http://localhost:8000/docs`

---

## 📂 Project Structure

```
SoftAI---DataForLife---MedSightAI/
│
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── api/v1/endpoints/        # API route handlers
│   │   │   ├── patients.py          # Patient CRUD
│   │   │   ├── cases.py             # Case management
│   │   │   ├── analysis.py          # AI analysis
│   │   │   ├── similarity.py        # Similar case search
│   │   │   ├── education.py         # Education mode
│   │   │   ├── reports.py           # Report generation
│   │   │   └── health.py            # Health checks
│   │   ├── config/                  # Configuration
│   │   │   ├── settings.py          # Environment settings
│   │   │   ├── database.py          # PostgreSQL config
│   │   │   ├── s3.py                # AWS S3 config
│   │   │   └── zilliz.py            # Zilliz Cloud config
│   │   ├── models/                  # SQLAlchemy models
│   │   │   └── models.py            # Database schema
│   │   ├── schemas/                 # Pydantic schemas
│   │   │   ├── patient.py           # Patient validation
│   │   │   ├── case.py              # Case validation
│   │   │   ├── ai_result.py         # AI result schemas
│   │   │   └── ...
│   │   ├── core/                    # Business logic
│   │   │   ├── crud_patient.py      # Patient operations
│   │   │   ├── crud_case.py         # Case operations
│   │   │   └── ...
│   │   ├── services/                # External services
│   │   │   ├── ai_service.py        # AI model integration
│   │   │   ├── llm_service.py       # MedGemma LLM
│   │   │   ├── s3_service.py        # S3 operations
│   │   │   └── zilliz_service.py    # Vector search
│   │   └── utils/                   # Utilities
│   │       └── s3_paths.py          # S3 path builder
│   ├── main.py                      # FastAPI app entry
│   ├── requirements.txt             # Python dependencies
│   ├── .env.example                 # Environment template
│   └── README.md                    # Backend docs
│
├── frontend/                         # React + Electron Frontend
│   ├── src/
│   │   ├── components/              # React components
│   │   │   ├── Home/               # Landing page
│   │   │   ├── DoctorDetail/       # Doctor mode UI
│   │   │   ├── StudentDetail/      # Education mode UI
│   │   │   ├── custom/             # Shared components
│   │   │   └── layout/             # Layout components
│   │   ├── constants/              # Constants and configs
│   │   ├── routes/                 # React Router pages
│   │   ├── main.js                 # Electron main process
│   │   ├── preload.js              # Electron preload
│   │   └── renderer.jsx            # React entry point
│   ├── package.json                # Node dependencies
│   └── vite.config.mjs             # Vite configuration
│
├── MedSightAI/                      # AI Model (DenseNet121/ViT)
│   ├── inference.py                # Model inference
│   ├── train.py                    # Model training
│   └── src/
│       ├── model.py                # Model architecture
│       ├── dataset.py              # Data preprocessing
│       └── loss.py                 # Loss functions
│
├── medgemma/                        # MedGemma LLM Integration
│   ├── generate_report.py         # Report generation
│   ├── test_local.py              # Local testing
│   └── README.md                  # MedGemma docs
│
├── VindrDataset/                   # VinDr Dataset Processing
│   ├── train.py                   # Training scripts
│   └── src/                       # Dataset utilities
│
└── README.md                       # This file
```

---

## 💻 Development

### Backend Development

**Run tests**
```bash
cd backend
pytest
```

**Test individual services**
```bash
python test_db_connection.py      # Test PostgreSQL
python test_s3_connection.py       # Test AWS S3
python test_zilliz_service.py      # Test Zilliz Cloud
```

**Check API errors**
```bash
curl http://localhost:8000/api/v1/health
```

**Format code**
```bash
black app/
```

### Frontend Development

**Run development server**
```bash
cd frontend
npm run start
```

**Build for production**
```bash
npm run make
```

**Package application**
```bash
npm run package
```

### Database Migrations

Using Alembic for database migrations:
```bash
cd backend
alembic revision --autogenerate -m "Add new table"
alembic upgrade head
```

---

## 📊 Database Schema

### PostgreSQL Tables

1. **patient** - Patient information
   - id (UUID, PK)
   - name, age, gender
   - history (JSONB) - symptoms, medical history, test results
   - created_at (timestamp)

2. **cases** - X-ray cases
   - id (UUID, PK)
   - patient_id (FK → patient)
   - image_path (S3 path)
   - processed_img_path (S3 path)
   - similar_cases (JSON array)
   - similarity_scores (JSON array)
   - timestamp

3. **ai_result** - AI analysis results
   - id (UUID, PK)
   - case_id (FK → cases)
   - predictions (JSONB)
   - bounding_boxes (JSONB)
   - gradcam_path (S3 path)
   - confidence_scores (JSONB)
   - concepts (JSONB)
   - created_at

4. **report** - Medical reports
   - id (UUID, PK)
   - case_id (FK → cases)
   - model_report (Text) - AI-generated
   - doctor_report (Text) - Doctor's final report
   - status (Enum)
   - created_at, updated_at

5. **chat_session** - Education mode chat sessions
   - id (UUID, PK)
   - user_id (UUID)
   - case_id (FK → cases)
   - session_type (Enum: practice, tutoring)
   - score (Float)
   - started_at, ended_at

6. **chat_message** - Chat messages
   - id (UUID, PK)
   - session_id (FK → chat_session)
   - sender (Enum: user, ai)
   - message (Text)
   - timestamp

### Zilliz Cloud Collection

**med_vector** - Vector embeddings for similarity search
- primary_key (int64) - Mapped from Case UUID
- txt_emb (float vector[1152]) - Text embedding from MedSigLip
- img_emb (float vector[1152]) - Image embedding from MedSigLip

### AWS S3 Structure

```
aithena/
├── patients/{patient_id}/profile/
├── cases/{case_id}/
│   ├── original/           # Original X-ray images
│   ├── processed/          # Preprocessed images
│   ├── annotated/          # Grad-CAM, bounding boxes
│   ├── segmentation/       # Segmentation masks
│   └── reports/            # PDF reports
├── education/{session_id}/
│   ├── student_uploads/
│   ├── student_annotations/
│   └── feedback/
└── similar_cases/{case_id}/thumbnails/
```

---

## 🎯 Project Objectives

### General Objectives

1. **Assist Doctors**
   - Provide visual suggestions and lesion areas identified by AI
   - Support diagnosis of lung diseases, particularly tuberculosis
   - Enhance diagnostic accuracy with similar case references

2. **Medical Education**
   - Create learning environment for medical students
   - Enable practice with immediate feedback
   - Compare student analysis with AI results
   - Explain model's decision-making mechanism

3. **Data Platform**
   - Build framework for collecting X-ray data
   - Expand and standardize medical image database
   - Support future AI model development for various diseases

### Technical Objectives

1. **Deep Learning Models**
   - Train CNN/ViT models for lung abnormality detection
   - Classify diseases based on image features
   - Optimize for edge devices and low-tier hospital systems

2. **Explainable AI**
   - Use Grad-CAM to highlight lesion areas
   - Map features to medical concepts (tissue damage, opacity, effusion)
   - Provide transparent, verifiable explanations

3. **Knowledge Integration**
   - Build Knowledge Graph with Vision Transformer
   - Generate automated preliminary medical reports
   - Enhance diagnostic reliability with medical reasoning

4. **Data Pipeline**
   - Automate data collection and labeling
   - Facilitate continuous model training
   - Support incremental learning from doctor feedback

---

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/
```

### Service Tests

```bash
# Test Zilliz vector database
python test_zilliz_service.py

# Test S3 storage
python test_s3_connection.py

# Test database connection
python test_db_connection.py
```

### API Tests

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Create patient
curl -X POST http://localhost:8000/api/v1/patients/ \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "age": 45, "gender": "male"}'
```

---

## 🚢 Deployment

### Docker Deployment (Optional)

```bash
cd backend
docker build -t medsight-backend .
docker run -p 8000:8000 --env-file .env medsight-backend
```

### Production Deployment

1. Set up production database (Neon PostgreSQL)
2. Configure AWS S3 bucket
3. Set up Zilliz Cloud cluster
4. Update `.env` with production credentials
5. Deploy backend to cloud platform (AWS, Azure, GCP)
6. Build and distribute Electron app

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 👥 Team

**SoftAI Development Team**

- **Project Lead**: [Team Lead Name]
- **Backend Developer**: [Backend Dev Name]
- **Frontend Developer**: NhanPhamThanh-IT
- **AI/ML Engineer**: [ML Engineer Name]
- **Medical Advisor**: [Medical Expert Name]

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Contact

For questions or support:
- **Email**: [team@softai.com]
- **GitHub**: [SoftAI Repository](https://github.com/Tuprott991/SoftAI---DataForLife---MedSightAI)

---

## 🙏 Acknowledgments

- VinDr Dataset for medical imaging data
- MedGemma team for medical LLM
- Zilliz Cloud for vector database infrastructure
- Neon for PostgreSQL cloud database
- AWS for S3 storage infrastructure

---

<div align="center">

**Built with ❤️ by SoftAI Team**

</div>
