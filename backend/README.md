# MedSight AI Backend

FastAPI backend for MedSight AI - Medical Imaging Analysis Platform with Explainable AI

## 🏗️ Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/      # API route handlers
│   │           ├── patients.py
│   │           ├── cases.py
│   │           ├── analysis.py
│   │           ├── reports.py
│   │           ├── education.py
│   │           ├── similarity.py
│   │           └── health.py
│   ├── config/                 # Configuration files
│   │   ├── settings.py        # Environment settings
│   │   ├── database.py        # Database connection
│   │   ├── s3.py             # AWS S3 config
│   │   └── zilliz.py         # Zilliz Cloud vector DB config
│   ├── models/                # SQLAlchemy models
│   │   └── models.py
│   ├── schemas/               # Pydantic schemas
│   │   ├── patient.py
│   │   ├── case.py
│   │   ├── ai_result.py
│   │   ├── report.py
│   │   ├── chat.py
│   │   └── common.py
│   ├── core/                  # Business logic & CRUD
│   │   ├── crud.py
│   │   ├── crud_patient.py
│   │   ├── crud_case.py
│   │   ├── crud_ai_result.py
│   │   ├── crud_report.py
│   │   └── crud_chat.py
│   ├── services/              # External services integration
│   │   ├── s3_service.py      # S3 operations
│   │   ├── zilliz_service.py  # Vector search (Zilliz Cloud)
│   │   ├── ai_service.py      # AI model integration (TODO)
│   │   └── llm_service.py     # MedGemma integration (TODO)
│   ├── utils/                 # Utility functions
│   │   └── helpers.py
│   └── middleware/            # Middleware
│       └── cors.py
├── main.py                    # FastAPI app entry point
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── .env                       # Environment variables (DO NOT COMMIT)
└── .env.example              # Environment template

```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL (Neon)
- AWS S3 bucket
- Zilliz Cloud account (vector database)

### Installation

1. **Clone the repository** (if not already done)

2. **Navigate to backend directory**
```bash
cd backend
```

3. **Create virtual environment**
```bash
python -m venv venv
```

4. **Activate virtual environment**
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

5. **Install dependencies**
```bash
pip install -r requirements.txt
```

6. **Setup environment variables**
```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your credentials
# - DATABASE_URL: Your Neon PostgreSQL connection string
# - AWS credentials: Access key, secret key, bucket name
# - ZILLIZ_CLOUD_URI: Zilliz Cloud serverless endpoint
# - ZILLIZ_CLOUD_API_KEY: Your Zilliz API key
```

7. **Initialize Neon PostgreSQL schema**
```bash
python scripts/init_neon_schema.py
```

8. **Create Zilliz collection (`med_vector`)**
```bash
python scripts/create_zilliz_collection.py
```

### One-command datastore bootstrap

If you want a quick setup that matches the target schema for PostgreSQL and Zilliz Cloud, use:

```bash
python scripts/bootstrap_datastores.py
```

This script:

- creates PostgreSQL enums, tables, indexes, and the `report.updated_at` trigger
- creates the Zilliz Cloud `med_vector` collection if it does not already exist
- only requires datastore variables from `.env`

See [`scripts/SETUP_GUIDE.md`](/mnt/c/Users/HP/Coding/MedSightAI-DataForLife/backend/scripts/SETUP_GUIDE.md) for the exact schema assumptions and the manual Neon/Zilliz console steps.

### Configuration

Edit `.env` file with your credentials:

```env
# Database
DATABASE_URL=postgresql+psycopg2://user:pass@your-neon-host:5432/medsight_db?sslmode=require

# AWS S3
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET_NAME=your-bucket

# Zilliz Cloud Vector Database
ZILLIZ_CLOUD_URI=https://your-cluster.cloud.zilliz.com
ZILLIZ_CLOUD_API_KEY=your_api_key_here
ZILLIZ_COLLECTION_NAME=med_vector
ZILLIZ_TXT_DIMENSION=1152
ZILLIZ_IMG_DIMENSION=1152
```

### Datastore Setup Notes

- Neon table schema is created from SQLAlchemy models in `app/models/models.py`.
- The backend startup (`main.py`) also runs table creation, but running `scripts/init_neon_schema.py` first gives you an explicit setup step.
- Zilliz collection expected by this project:
	- `primary_key` (`Int64`, primary key, no auto-id)
	- `txt_emb` (`FloatVector`, 1152)
	- `img_emb` (`FloatVector`, 1152)

### Running the Application

**Option 1: Direct Python**
```bash
python main.py
```

**Option 2: Uvicorn**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Option 3: Docker Compose**
```bash
docker-compose up -d
```

The API will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

## 📚 API Endpoints

### Core Endpoints

- **Health Check**: `GET /api/v1/health`
- **Version**: `GET /api/v1/version`

### Patient Management
- `POST /api/v1/patients` - Create patient
- `GET /api/v1/patients` - List patients
- `GET /api/v1/patients/{id}` - Get patient
- `PUT /api/v1/patients/{id}` - Update patient
- `DELETE /api/v1/patients/{id}` - Delete patient

### Case Management
- `POST /api/v1/cases` - Upload case with X-ray
- `GET /api/v1/cases` - List cases
- `GET /api/v1/cases/{id}` - Get case
- `PUT /api/v1/cases/{id}` - Update case
- `DELETE /api/v1/cases/{id}` - Delete case
- `GET /api/v1/cases/{id}/image-url` - Get presigned image URL

### AI Analysis
- `POST /api/v1/analysis/full-pipeline` - Run complete analysis
- `POST /api/v1/analysis/preprocess` - Preprocess image
- `POST /api/v1/analysis/inference` - Run model inference
- `GET /api/v1/analysis/{case_id}/heatmap` - Get Grad-CAM heatmap
- `GET /api/v1/analysis/{case_id}/concepts` - Get concept analysis

### Reports
- `POST /api/v1/reports/generate` - Generate AI report
- `GET /api/v1/reports/{case_id}` - Get report
- `PUT /api/v1/reports/{id}/doctor-report` - Update doctor's report
- `PUT /api/v1/reports/{id}/feedback` - Add feedback

### Education Mode
- `GET /api/v1/education/practice-cases` - Get practice cases
- `POST /api/v1/education/submit-answer` - Submit student answer
- `POST /api/v1/education/sessions` - Create chat session
- `GET /api/v1/education/sessions` - List sessions
- `POST /api/v1/education/sessions/{id}/messages` - Send message

### Similarity Search
- `POST /api/v1/similarity/search` - Search similar cases
- `POST /api/v1/similarity/embed` - Generate embeddings

## 🔧 Integration with AI Models

### TODO: Connect to Other Modules

The backend has placeholder services that need to be connected to AI models:

**1. AI Model Service** (`app/services/ai_service.py`)
- Connect to `MedSightAI/inference.py` for model inference
- Connect to preprocessing functions
- Implement Grad-CAM generation
- Implement concept extraction

**2. MedGemma Service** (`app/services/llm_service.py`)
- Connect to `medgemma/generate_report.py`
- Implement report generation
- Implement educational chat

**3. MedSigLip Service** (`app/services/ai_service.py`)
- Implement image embedding generation
- Implement text embedding generation

## 🗄️ Database Schema

Tables managed by SQLAlchemy:
- `patient` - Patient information
- `cases` - Medical imaging cases
- `ai_result` - AI analysis results
- `report` - Medical reports
- `chat_session` - Education chat sessions
- `chat_message` - Chat messages

## 🧪 Testing

```bash
pytest
```

## 📦 Deployment

### Using Docker

```bash
# Build image
docker build -t medsight-backend .

# Run container
docker run -p 8000:8000 --env-file .env medsight-backend
```

### Using Docker Compose

```bash
docker-compose up -d
```

## 🛠️ Development

### Code Style

```bash
black app/
```

### Adding New Endpoints

1. Create route handler in `app/api/v1/endpoints/`
2. Add CRUD operations in `app/core/`
3. Define schemas in `app/schemas/`
4. Register router in `app/api/v1/__init__.py`

## 📝 Environment Variables

See `.env.example` for all required environment variables.

## 🤝 Contributing

This backend is part of the MedSight AI project and integrates with:
- `MedSightAI/` - AI model inference
- `medgemma/` - Medical report generation
- `VindrDataset/` - Dataset handling

## 📄 License

[Your License]

## 👥 Team

Backend Engineering Team
