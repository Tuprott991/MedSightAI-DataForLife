import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { patientsData } from '../constants/patients';
import {
    ImageInteractiveSection,
    SubmitSection,
    ChatbotSection
} from '../components/StudentDetail';
import { Toast } from '../components/custom/Toast';
import { User, Calendar } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const StudentDetail = () => {
    const { t } = useTranslation();
    const { id } = useParams();
    const patient = patientsData.find(p => p.id === parseInt(id));
    const [caseData, setCaseData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [annotations, setAnnotations] = useState([]);
    const [submissionData, setSubmissionData] = useState(null);
    const [toast, setToast] = useState(null);

    // Status color helper function (matching PatientCard)
    const getStatusColor = (status) => {
        switch (status) {
            case 'Critical':
                return 'bg-red-600/30 text-white border-red-600/40';
            case 'Under Treatment':
                return 'bg-blue-500/30 text-white border-blue-500/40';
            case 'Stable':
                return 'bg-green-500/30 text-white border-green-500/40';
            case 'Admitted':
                return 'bg-teal-500/30 text-white border-teal-500/40';
            default:
                return 'bg-gray-500/30 text-white border-gray-500/40';
        }
    };

    const getStatusText = (status) => {
        const statusMap = {
            'Critical': t('doctor.patientCard.statusCritical'),
            'Under Treatment': t('doctor.patientCard.statusUnderTreatment'),
            'Stable': t('doctor.patientCard.statusStable'),
            'Admitted': t('doctor.patientCard.statusAdmitted')
        };
        return statusMap[status] || status;
    };

    // Simulate API call to fetch case data
    useEffect(() => {
        const fetchCaseData = async () => {
            setLoading(true);

            // Mock API delay
            await new Promise(resolve => setTimeout(resolve, 800));

            // Mock case data
            const mockCase = {
                id: parseInt(id),
                patientName: patient?.name || 'Unknown Patient',
                age: patient?.age || 0,
                gender: patient?.gender || 'Unknown',
                diagnosis: patient?.diagnosis || 'Unknown Condition',
                imageUrl: patient?.image || '',
                description: 'Medical imaging case for educational purposes',
                difficulty: 'Intermediate'
            };

            setCaseData(mockCase);
            setLoading(false);
        };

        fetchCaseData();
    }, [id, patient]);

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

    if (!patient) {
        return (
            <div className="h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold mb-2">{t('doctor.noResults')}</h1>
                    <p className="text-gray-400">{t('common.error')}</p>
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
