import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { Sparkles } from 'lucide-react';
import { patientsData } from '../constants/patients';
import { medicalImagesGroups, generateAnalysisData } from '../constants/medicalData';
import {
    ImageListGrouped,
    ImageViewer
} from '../components/DoctorDetail';
import { AnalysisTab, RecommendationsTab } from '../components/DoctorDetail/tabs';
import { useSidebar } from '../components/layout';

export const DoctorDetail = () => {
    const { id } = useParams();
    const patient = patientsData.find(p => p.id === parseInt(id));
    const { isLeftCollapsed } = useSidebar();

    // Set first image from first group as default
    const firstImage = medicalImagesGroups[0]?.images[0];
    const [selectedImage, setSelectedImage] = useState(firstImage);
    const [analysisData, setAnalysisData] = useState(null);

    const handleAIAnalyze = () => {
        // TODO: Replace with actual API call
        // For now, generate mock analysis data
        const data = patient ? generateAnalysisData(patient.diagnosis) : null;
        setAnalysisData(data);
    };

    if (!patient) {
        return (
            <div className="min-h-screen bg-[#0a0a0a] text-white">
                <div className="container mx-auto px-6 py-8">
                    <div className="text-center py-20">
                        <h1 className="text-2xl font-bold mb-4">Patient Not Found</h1>
                        <p className="text-gray-400 text-sm">The patient record you're looking for doesn't exist.</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-[#0a0a0a] text-white">
            <div className="container mx-auto px-6 pt-6 pb-4">
                {/* Three Column Layout: 2:3:2 ratio (responsive based on collapse) */}
                <div className={`grid grid-cols-1 gap-4 ${isLeftCollapsed
                    ? 'lg:grid-cols-6'
                    : 'lg:grid-cols-7'
                    }`}>

                    {/* Left Column - Image List (2/7 or hidden when collapsed) */}
                    {!isLeftCollapsed && (
                        <div className="lg:col-span-2">
                            <ImageListGrouped
                                imageGroups={medicalImagesGroups}
                                selectedImage={selectedImage}
                                onImageSelect={setSelectedImage}
                                patient={patient}
                            />
                        </div>
                    )}

                    {/* Middle Column - Selected Image (3/7 or expanded to 4/6) */}
                    <div className={
                        isLeftCollapsed
                            ? 'lg:col-span-4'
                            : 'lg:col-span-3'
                    }>
                        <ImageViewer
                            image={selectedImage}
                            patientInfo={patient}
                        />
                    </div>

                    {/* Right Column - AI Analysis (2/7) */}
                    <div className="lg:col-span-2">
                        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden h-[calc(100vh-110px)] flex flex-col">
                            {/* Header with AI Analyze Button */}
                            <div className="px-4 py-3 border-b border-white/10 bg-[#141414] flex items-center justify-between">
                                <h3 className="text-base font-semibold text-white">Reporting</h3>

                                {/* AI Analyze Button - Always Teal */}
                                <button
                                    onClick={handleAIAnalyze}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg transition-all bg-amber-500 text-white shadow-lg shadow-amber-500/50 hover:bg-amber-600 active:scale-95 cursor-pointer"
                                >
                                    <Sparkles className="w-3.5 h-3.5" />
                                    <span className="font-medium">AI Analyze</span>
                                </button>
                            </div>

                            {/* Scrollable Content */}
                            <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
                                {analysisData ? (
                                    <div className="space-y-4">
                                        <AnalysisTab
                                            findings={analysisData.findings}
                                            metrics={analysisData.metrics}
                                        />
                                        <div className="border-t border-white/10 pt-4">
                                            <RecommendationsTab
                                                recommendations={analysisData.recommendations}
                                            />
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex items-center justify-center h-full text-center">
                                        <div>
                                            <Sparkles className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                                            <p className="text-sm text-gray-500">Click "AI Analyze" to generate</p>
                                            <p className="text-sm text-gray-500">automated analysis</p>
                                        </div>
                                    </div>
                                )}
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
