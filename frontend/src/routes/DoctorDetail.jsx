import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { Sparkles, RefreshCw } from 'lucide-react';
import { patientsData } from '../constants/patients';
import { medicalImagesGroups, generateAnalysisData } from '../constants/medicalData';
import {
    ImageListGrouped,
    ImageViewer
} from '../components/DoctorDetail';
import { AnalysisTab, RecommendationsTab } from '../components/DoctorDetail/AnalysisTab';
import { useSidebar } from '../components/layout';

export const DoctorDetail = () => {
    const { id } = useParams();
    const patient = patientsData.find(p => p.id === parseInt(id));
    const { isLeftCollapsed } = useSidebar();

    // Set first image from first group as default
    const firstImage = medicalImagesGroups[0]?.images[0];
    const [selectedImage, setSelectedImage] = useState(firstImage);
    const [originalImage, setOriginalImage] = useState(firstImage);
    const [analysisData, setAnalysisData] = useState(null);
    const [isUpdating, setIsUpdating] = useState(false);
    const [selectedFindingIds, setSelectedFindingIds] = useState([]);

    const handleAIAnalyze = () => {
        // TODO: Replace with actual API call
        // For now, generate mock analysis data
        const data = patient ? generateAnalysisData(patient.diagnosis) : null;
        setAnalysisData(data);
    };

    const handleFindingSelectionChange = (selectedIds) => {
        setSelectedFindingIds(selectedIds);
    };

    const handleUpdateClick = async () => {
        if (!analysisData) return;

        setIsUpdating(true);

        // Mock API delay
        setTimeout(() => {
            // 3 kịch bản ngẫu nhiên
            const scenarios = [
                // Kịch bản 1: Tình trạng nghiêm trọng
                [
                    { name: 'Viêm phổi nặng', confidence: 88 },
                    { name: 'Tràn dịch màng phổi', confidence: 75 },
                    { name: 'Suy hô hấp cấp', confidence: 68 }
                ],
                // Kịch bản 2: Tình trạng trung bình
                [
                    { name: 'Viêm phế quản mãn tính', confidence: 72 },
                    { name: 'Hen phế quản', confidence: 65 },
                    { name: 'Viêm phổi nhẹ', confidence: 58 }
                ],
                // Kịch bản 3: Tình trạng nhẹ/khác
                [
                    { name: 'Viêm đường hô hấp trên', confidence: 82 },
                    { name: 'Cảm cúm thông thường', confidence: 70 },
                    { name: 'Dị ứng đường hô hấp', confidence: 55 }
                ]
            ];

            // Chọn ngẫu nhiên 1 trong 3 kịch bản
            const randomScenario = scenarios[Math.floor(Math.random() * scenarios.length)];

            setAnalysisData({
                ...analysisData,
                suspectedDiseases: randomScenario
            });

            setIsUpdating(false);
        }, 1500);
    };

    const handleFindingClick = (imageData) => {
        // Update the selected image when a finding is clicked
        setSelectedImage(imageData);
    };

    const handleRestoreOriginal = () => {
        // Restore to original single image
        setSelectedImage(originalImage);
    };

    const handleImageSelect = (image) => {
        // When manually selecting from list, update both selected and original
        setSelectedImage(image);
        setOriginalImage(image);
    };

    if (!patient) {
        return (
            <div className="min-h-screen bg-[#0a0a0a] text-white">
                <div className="container mx-auto px-6 py-8">
                    <div className="text-center py-20">
                        <h1 className="text-2xl font-bold mb-4">Không Tìm Thấy Bệnh Nhân</h1>
                        <p className="text-gray-400 text-sm">Hồ sơ bệnh nhân bạn tìm kiếm không tồn tại.</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-[#0a0a0a] text-white">
            <div className="container mx-auto px-6 pt-6 pb-4">
                {/* Three Column Layout: 1.8:4.2:2 ratio (responsive based on collapse) */}
                <div className={`grid grid-cols-1 gap-4 ${isLeftCollapsed
                    ? 'lg:grid-cols-8'
                    : 'lg:grid-cols-8'
                    }`}>

                    {/* Left Column - Image List (2/8 or hidden when collapsed) */}
                    {!isLeftCollapsed && (
                        <div className="lg:col-span-2">
                            <ImageListGrouped
                                imageGroups={medicalImagesGroups}
                                selectedImage={selectedImage}
                                onImageSelect={handleImageSelect}
                                patient={patient}
                            />
                        </div>
                    )}

                    {/* Middle Column - Selected Image (4/8 or expanded to 6/8) */}
                    <div className={
                        isLeftCollapsed
                            ? 'lg:col-span-6'
                            : 'lg:col-span-4'
                    }>
                        <ImageViewer
                            image={selectedImage}
                            patientInfo={patient}
                            onRestoreOriginal={handleRestoreOriginal}
                        />
                    </div>

                    {/* Right Column - AI Analysis (2/8) */}
                    <div className="lg:col-span-2">
                        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden h-[calc(100vh-110px)] flex flex-col">
                            {/* Header with AI Analyze Button */}
                            <div className="px-4 py-3 border-b border-white/10 bg-[#141414] flex items-center justify-between">
                                <h3 className="text-base font-semibold text-white">Báo Cáo</h3>
                                {/* AI Analyze Button */}
                                <button
                                    onClick={handleAIAnalyze}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg transition-all bg-amber-500 text-white shadow-lg shadow-amber-500/50 hover:bg-amber-600 active:scale-95 cursor-pointer"
                                >
                                    <Sparkles className="w-3.5 h-3.5" />
                                    <span className="font-medium">Phân tích</span>
                                </button>
                            </div>

                            {/* Scrollable Content */}
                            <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
                                {analysisData ? (
                                    <div className="space-y-4">
                                        <AnalysisTab
                                            findings={analysisData.findings}
                                            suspectedDiseases={analysisData.suspectedDiseases}
                                            onFindingClick={handleFindingClick}
                                            onFindingSelectionChange={handleFindingSelectionChange}
                                            onUpdateClick={handleUpdateClick}
                                            isUpdating={isUpdating}
                                            selectedFindingIds={selectedFindingIds}
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
                                            <p className="text-sm text-gray-500">Nhấp "Phân Tích AI" để tạo</p>
                                            <p className="text-sm text-gray-500">phân tích tự động</p>
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
