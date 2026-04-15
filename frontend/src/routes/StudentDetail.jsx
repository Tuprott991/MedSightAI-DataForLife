import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { getPatientDetail } from '../services/patientApi';
import {
    ImageInteractiveSection,
    SubmitSection,
    ChatbotSection
} from '../components/StudentDetail';
import { Toast } from '../components/custom/Toast';
import { User, Calendar, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const StudentDetail = () => {
    const { t } = useTranslation();
    const { id } = useParams();
    const [patient, setPatient] = useState(null);
    const [caseData, setCaseData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [annotations, setAnnotations] = useState([]);
    const [submissionData, setSubmissionData] = useState(null);
    const [toast, setToast] = useState(null);

    // Status color helper function (matching PatientCard)
    const getStatusColor = (status) => {
        const statusLower = status?.toLowerCase();
        switch (statusLower) {
            case 'critical':
                return 'bg-red-600/30 text-white border-red-600/40';
            case 'improving':
                return 'bg-blue-500/30 text-white border-blue-500/40';
            case 'stable':
                return 'bg-green-500/30 text-white border-green-500/40';
            case 'admitted':
                return 'bg-teal-500/30 text-white border-teal-500/40';
            default:
                return 'bg-gray-500/30 text-white border-gray-500/40';
        }
    };

    const getStatusText = (status) => {
        const statusLower = status?.toLowerCase();
        const statusMap = {
            'critical': t('doctor.patientCard.statusCritical'),
            'improving': t('doctor.patientCard.statusUnderTreatment'),
            'stable': t('doctor.patientCard.statusStable'),
            'admitted': t('doctor.patientCard.statusAdmitted')
        };
        return statusMap[statusLower] || status;
    };

    // Fetch patient data from API
    useEffect(() => {
        const fetchCaseData = async () => {
            setLoading(true);
            setError(null);
            
            try {
                const data = await getPatientDetail(id);
                setPatient(data);
                
                // For students, anonymize patient information
                const mockCase = {
                    id: data.id,
                    caseId: data.latest_case?.id || null,
                    patientName: `Case ${data.id.substring(0, 8)}`, // Anonymize
                    age: null, // Hide age
                    gender: null, // Hide gender
                    diagnosis: null, // Hide diagnosis
                    imageUrl: data.latest_case?.image_path || data.latest_case?.processed_img_path || '',
                    description: 'Medical imaging case for educational purposes',
                    difficulty: 'Intermediate',
                    status: data.status
                };

                setCaseData(mockCase);
            } catch (err) {
                console.error('Error fetching case data:', err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        if (id) {
            fetchCaseData();
        }
    }, [id]);

    const handleAnnotationsChange = (newAnnotations) => {
        setAnnotations(newAnnotations);
    };

    const handleSubmitDiagnosis = (submissionData) => {
        console.log('Student submitted diagnosis:', submissionData);
        console.log('Annotations:', submissionData.annotations);

        // Lưu submission data để trigger chatbot analysis
        setSubmissionData(submissionData);

        // TODO: Send to API
        return { success: true };
    };

    const showToast = (type, message) => {
        setToast({ type, message });
    };

    if (loading) {
        return (
            <div className="h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
                <Loader2 className="w-12 h-12 text-teal-500 animate-spin" />
            </div>
        );
    }

    if (error || !patient || !caseData) {
        return (
            <div className="h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold mb-2">{t('doctor.noResults')}</h1>
                    <p className="text-gray-400">{error || t('common.error')}</p>
                </div>
            </div>
        );
    }

    if (loading) {
        return (
            <div className="h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
                <div className="text-center">
                    <div className="w-12 h-12 border-4 border-teal-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-gray-400">{t('common.loading')}</p>
                </div>
            </div>
        );
    }

    return (
        <div className="h-[91vh] bg-[#0a0a0a] text-white overflow-hidden flex flex-col">
            {/* Toast Notification */}
            {toast && (
                <Toast
                    type={toast.type}
                    message={toast.message}
                    onClose={() => setToast(null)}
                />
            )}
            {/* Main Content - Fixed Height */}
            <div className="flex-1 overflow-hidden min-h-0">
                <div className="h-full px-6 py-6">
                    {/* Two Column Layout */}
                    <div className="grid grid-cols-1 lg:grid-cols-7 gap-6 h-full max-w-[1600px] mx-auto">

                        {/* Left Column - Image + Submit (5/7) */}
                        <div className="lg:col-span-5 flex flex-col gap-4 h-full min-h-0">
                            {/* Image Interactive Section - Takes remaining space */}
                            <ImageInteractiveSection
                                caseData={caseData}
                                onAnnotationsChange={handleAnnotationsChange}
                            />

                            {/* Submit Section - Fixed height */}
                            <SubmitSection
                                onSubmit={handleSubmitDiagnosis}
                                annotations={annotations}
                                showToast={showToast}
                            />
                        </div>

                        {/* Right Column - Chatbot (2/7) */}
                        <div className="lg:col-span-2 h-full min-h-0">
                            <ChatbotSection
                                annotations={annotations}
                                caseData={caseData}
                                submissionData={submissionData}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StudentDetail;
