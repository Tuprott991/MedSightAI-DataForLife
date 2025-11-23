import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { patientsData } from '../constants/patients';
import {
    ImageInteractiveSection,
    SubmitSection,
    ChatbotSection
} from '../components/StudentDetail';
import { Toast } from '../components/custom/Toast';

export const StudentDetail = () => {
    const { id } = useParams();
    const patient = patientsData.find(p => p.id === parseInt(id));
    const [caseData, setCaseData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [annotations, setAnnotations] = useState([]);
    const [toast, setToast] = useState(null);

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
                imageUrl: patient?.image || 'https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=800',
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
                    <h1 className="text-2xl font-bold mb-2">Case Not Found</h1>
                    <p className="text-gray-400">The case you're looking for doesn't exist.</p>
                </div>
            </div>
        );
    }

    if (loading) {
        return (
            <div className="h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
                <div className="text-center">
                    <div className="w-12 h-12 border-4 border-teal-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-gray-400">Loading case...</p>
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
                            <ChatbotSection />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StudentDetail;
