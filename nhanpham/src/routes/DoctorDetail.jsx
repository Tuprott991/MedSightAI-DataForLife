import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { MoreVertical } from 'lucide-react';
import { patientsData } from '../constants/patients';
import { medicalImagesGroups, generateAnalysisData } from '../constants/medicalData';
import {
    ImageListGrouped,
    ImageViewer
} from '../components/DoctorDetail';
import { AnalysisTab, RecommendationsTab } from '../components/DoctorDetail/tabs';

export const DoctorDetail = () => {
    const { id } = useParams();
    const patient = patientsData.find(p => p.id === parseInt(id));

    // Set first image from first group as default
    const firstImage = medicalImagesGroups[0]?.images[0];
    const [selectedImage, setSelectedImage] = useState(firstImage);
    const [activeTab, setActiveTab] = useState('analysis');
    const [dropdownOpen, setDropdownOpen] = useState(false);

    // Generate analysis data based on patient
    const analysisData = patient ? generateAnalysisData(patient.diagnosis) : null;

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
                {/* Three Column Layout: 2:3:2 ratio */}
                <div className="grid grid-cols-1 lg:grid-cols-7 gap-4">

                    {/* Left Column - Image List (2/7) */}
                    <div className="lg:col-span-2">
                        <ImageListGrouped
                            imageGroups={medicalImagesGroups}
                            selectedImage={selectedImage}
                            onImageSelect={setSelectedImage}
                            patient={patient}
                        />
                    </div>

                    {/* Middle Column - Selected Image (3/7) */}
                    <div className="lg:col-span-3">
                        <ImageViewer image={selectedImage} />
                    </div>

                    {/* Right Column - AI Analysis (2/7) */}
                    <div className="lg:col-span-2">
                        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden h-[calc(100vh-110px)] flex flex-col">
                            {/* Header with Dropdown */}
                            <div className="px-4 py-3 border-b border-white/10 bg-[#141414] flex items-center justify-between">
                                <h3 className="text-base font-semibold text-white">Reporting</h3>

                                {/* Dropdown Menu */}
                                <div className="relative">
                                    <button
                                        onClick={() => setDropdownOpen(!dropdownOpen)}
                                        className="p-1 hover:bg-white/10 rounded transition-colors"
                                    >
                                        <MoreVertical className="w-4 h-4 text-gray-400" />
                                    </button>

                                    {dropdownOpen && (
                                        <div className="absolute right-0 mt-2 w-48 bg-[#1a1a1a] border border-white/10 rounded-lg shadow-lg z-10">
                                            <button
                                                onClick={() => {
                                                    setActiveTab('analysis');
                                                    setDropdownOpen(false);
                                                }}
                                                className={`w-full text-left px-4 py-2 text-sm transition-colors ${activeTab === 'analysis'
                                                    ? 'bg-teal-500/20 text-teal-400'
                                                    : 'text-gray-300 hover:bg-white/5'
                                                    }`}
                                            >
                                                Analysis
                                            </button>
                                            <button
                                                onClick={() => {
                                                    setActiveTab('recommendations');
                                                    setDropdownOpen(false);
                                                }}
                                                className={`w-full text-left px-4 py-2 text-sm transition-colors rounded-b-lg ${activeTab === 'recommendations'
                                                    ? 'bg-teal-500/20 text-teal-400'
                                                    : 'text-gray-300 hover:bg-white/5'
                                                    }`}
                                            >
                                                Recommendations
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Scrollable Content */}
                            <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
                                {activeTab === 'analysis' && (
                                    <AnalysisTab
                                        findings={analysisData.findings}
                                        metrics={analysisData.metrics}
                                    />
                                )}
                                {activeTab === 'recommendations' && (
                                    <RecommendationsTab
                                        recommendations={analysisData.recommendations}
                                    />
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
