import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { patientsData } from '../constants/patients';
import { medicalImagesGroups, generateAnalysisData } from '../constants/medicalData';
import {
    ImageListGrouped,
    ImageViewer,
    AIConfidence,
    KeyFindings,
    Measurements,
    Recommendations
} from '../components/DoctorDetail';

export const DoctorDetail = () => {
    const { id } = useParams();
    const patient = patientsData.find(p => p.id === parseInt(id));

    // Set first image from first group as default
    const firstImage = medicalImagesGroups[0]?.images[0];
    const [selectedImage, setSelectedImage] = useState(firstImage);

    // Generate analysis data based on patient
    const analysisData = patient ? generateAnalysisData(patient.diagnosis) : null;

    if (!patient) {
        return (
            <div className="min-h-screen bg-[#0a0a0a] text-white">
                <div className="container mx-auto px-6 py-8">
                    <Link
                        to="/doctor"
                        className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-6"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        <span className="text-sm">Back to Patient Records</span>
                    </Link>
                    <div className="text-center py-20">
                        <h1 className="text-2xl font-bold mb-4">Patient Not Found</h1>
                        <p className="text-gray-400 text-sm">The patient record you're looking for doesn't exist.</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-[#0a0a0a] text-white">
            <div className="container mx-auto px-6 py-6">
                {/* Back Button & Patient Info */}
                <div className="flex items-center justify-between mb-4">
                    <Link
                        to="/doctor"
                        className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        <span className="text-sm">Back to Patient Records</span>
                    </Link>
                    <div className="text-right">
                        <h2 className="text-lg font-semibold">{patient.name}</h2>
                        <p className="text-xs text-gray-400">{patient.diagnosis}</p>
                    </div>
                </div>

                {/* Three Column Layout: 2:3:2 ratio */}
                <div className="grid grid-cols-1 lg:grid-cols-7 gap-4">

                    {/* Left Column - Image List (2/7) */}
                    <div className="lg:col-span-2">
                        <ImageListGrouped
                            imageGroups={medicalImagesGroups}
                            selectedImage={selectedImage}
                            onImageSelect={setSelectedImage}
                        />
                    </div>

                    {/* Middle Column - Selected Image (3/7) */}
                    <div className="lg:col-span-3">
                        <ImageViewer image={selectedImage} />
                    </div>

                    {/* Right Column - AI Analysis (2/7) */}
                    <div className="lg:col-span-2">
                        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden h-[calc(100vh-180px)] flex flex-col">
                            {/* Header */}
                            <div className="px-4 py-3 border-b border-white/10 bg-[#141414]">
                                <h3 className="text-base font-semibold text-white">AI Analysis Report</h3>
                            </div>

                            {/* Scrollable Content */}
                            <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
                                <AIConfidence confidence={analysisData.aiConfidence} />
                                <KeyFindings findings={analysisData.findings} />
                                <Measurements metrics={analysisData.metrics} />
                                <Recommendations recommendations={analysisData.recommendations} />
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Custom Scrollbar Styles */}
            <style jsx global>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 6px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 3px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background: rgba(20, 184, 166, 0.3);
                    border-radius: 3px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background: rgba(20, 184, 166, 0.5);
                }
            `}</style>
        </div>
    );
};

export default DoctorDetail;
