import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { Sparkles, RefreshCw, Bot, FileText, Send, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { patientsData } from '../constants/patients';
import { medicalImagesGroups, generateAnalysisData, getFindingImagePath, getPrototypeImagePath } from '../constants/medicalData';
import { generatePatientReport } from '../constants/reportData';
import {
    ImageListGrouped,
    ImageViewer
} from '../components/DoctorDetail';
import { AnalysisTab, RecommendationsTab } from '../components/DoctorDetail/AnalysisTab';
import { ReportPDF } from '../components/DoctorDetail/ReportPDF';
import { useSidebar } from '../components/layout';

export const DoctorDetail = () => {
    const { t } = useTranslation();
    const { id } = useParams();
    const patient = patientsData.find(p => p.id === parseInt(id));
    const { isLeftCollapsed } = useSidebar();

    // Use patient's image as default
    const patientImageData = patient ? {
        id: 0,
        url: patient.image,
        type: "Ảnh X-quang chính",
        imageCode: `IMG-${patient.id}-MAIN`,
        modality: "X-Ray"
    } : medicalImagesGroups[0]?.images[0];

    const [selectedImage, setSelectedImage] = useState(patientImageData);
    const [originalImage, setOriginalImage] = useState(patientImageData);
    const [analysisData, setAnalysisData] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [isUpdating, setIsUpdating] = useState(false);
    const [selectedFindingIds, setSelectedFindingIds] = useState([]);
    const [selectedFindingId, setSelectedFindingId] = useState(null);
    const [showChatbot, setShowChatbot] = useState(false);
    const [isGeneratingReport, setIsGeneratingReport] = useState(false);
    const [showReportModal, setShowReportModal] = useState(false);
    const [reportData, setReportData] = useState(null);
    const [isSimilarCaseMode, setIsSimilarCaseMode] = useState(false);
    const [similarCaseData, setSimilarCaseData] = useState(null);
    const [isLoadingSimilarAnalysis, setIsLoadingSimilarAnalysis] = useState(false);

    const handleAIAnalyze = () => {
        setIsAnalyzing(true);

        // Exit similar case mode and restore to original single image
        setIsSimilarCaseMode(false);
        setSimilarCaseData(null);
        setIsLoadingSimilarAnalysis(false);
        setSelectedImage(originalImage);

        // Random delay từ 3-6 giây
        const randomDelay = Math.floor(Math.random() * (6000 - 3000 + 1)) + 3000;

        setTimeout(() => {
            // Generate mock analysis data
            const data = patient ? generateAnalysisData(patient.diagnosis, patient.image) : null;
            setAnalysisData(data);
            setIsAnalyzing(false);
        }, randomDelay);
    };

    const handleFindingSelectionChange = (selectedIds) => {
        setSelectedFindingIds(selectedIds);
    };

    const handleChatbotToggle = () => {
        setShowChatbot(!showChatbot);
    };

    const handleGenerateReport = () => {
        setIsGeneratingReport(true);

        // Random delay từ 6-9 giây
        const randomDelay = Math.floor(Math.random() * (9000 - 6000 + 1)) + 6000;

        setTimeout(() => {
            // Generate report data
            const report = generatePatientReport(patient);
            setReportData(report);
            setIsGeneratingReport(false);
            setShowReportModal(true);
        }, randomDelay);
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

    const handleFindingClick = (finding) => {
        // Set selected finding ID for highlighting
        setSelectedFindingId(finding.id);

        // Get the image paths for this finding
        const findingImagePath = getFindingImagePath(finding.text, patient.image);
        const prototypeImagePath = getPrototypeImagePath(finding.text, patient.image);

        if (findingImagePath) {
            // Create xAI image (left side) - AI-enhanced image with finding highlighted
            const xaiImage = {
                id: `xai-${finding.id}`,
                url: findingImagePath,
                type: `xAI: ${finding.text}`,
                imageCode: `XAI-${finding.id}`,
                modality: "AI-Enhanced"
            };

            // If in similar case mode, recreate right image from similarCaseData to use latest imageUrl
            if (isSimilarCaseMode && similarCaseData) {
                const rightImage = {
                    id: `similar-${similarCaseData.patientName}`,
                    url: similarCaseData.imageUrl,
                    type: `Ca tương tự: ${similarCaseData.patientName}`,
                    imageCode: `SIMILAR-CASE`,
                    modality: "Comparison"
                };
                setSelectedImage([xaiImage, rightImage]);
            } else {
                // Normal mode: create right side image data with both original and prototype
                const rightImage = {
                    id: `right-${finding.id}`,
                    original: {
                        url: patient.image,
                        type: `Original: Ảnh gốc`,
                    },
                    prototype: {
                        url: prototypeImagePath,
                        type: `Prototype: ${finding.text}`,
                    },
                    imageCode: `RIGHT-${finding.id}`,
                    modality: "Comparison"
                };
                setSelectedImage([xaiImage, rightImage]);
            }
        }
    };

    const handleRestoreOriginal = () => {
        // Restore to original single image
        setSelectedImage(originalImage);
        // Exit similar case mode
        setIsSimilarCaseMode(false);
        setSimilarCaseData(null);
        setIsLoadingSimilarAnalysis(false);
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
                        <h1 className="text-2xl font-bold mb-4">{t('doctor.noResults')}</h1>
                        <p className="text-gray-400 text-sm">{t('doctor.noResults')}</p>
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
                            onSimilarCaseModeChange={(mode, data) => {
                                if (mode && data) {
                                    // Start loading for 6-7 seconds
                                    setIsLoadingSimilarAnalysis(true);
                                    const loadingDelay = Math.floor(Math.random() * (7000 - 6000 + 1)) + 6000;

                                    setTimeout(() => {
                                        setIsLoadingSimilarAnalysis(false);
                                        setIsSimilarCaseMode(true);
                                        setSimilarCaseData(data);
                                    }, loadingDelay);
                                } else {
                                    setIsSimilarCaseMode(false);
                                    setSimilarCaseData(null);
                                }
                            }}
                            onSimilarCaseDataChange={setSimilarCaseData}
                            onImageChange={setSelectedImage}
                        />
                    </div>

                    {/* Right Column - AI Analysis (2/8) */}
                    <div className="lg:col-span-2">
                        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden h-[calc(100vh-110px)] flex flex-col">
                            {/* Header with Action Buttons */}
                            <div className="px-4 py-3 border-b border-white/10 bg-[#141414] flex items-center justify-between">
                                <h3 className="text-base font-semibold text-white">Báo Cáo</h3>

                                {/* Action Buttons */}
                                <div className="flex items-center gap-2">
                                    {/* Chatbot Button */}
                                    <button
                                        onClick={handleChatbotToggle}
                                        disabled={isSimilarCaseMode}
                                        className={`w-8 h-8 flex items-center justify-center rounded-full transition-all ${isSimilarCaseMode
                                                ? 'bg-teal-500/20 text-teal-400 border border-teal-500/30 cursor-not-allowed opacity-50'
                                                : showChatbot
                                                    ? 'bg-teal-500 text-white'
                                                    : 'bg-teal-500/20 text-teal-400 border border-teal-500/30 hover:bg-teal-500 hover:text-white active:scale-95 cursor-pointer'
                                            }`}
                                        title={isSimilarCaseMode ? "Không khả dụng khi so sánh ca bệnh" : "Chatbot AI"}
                                    >
                                        <Bot className="w-4 h-4" />
                                    </button>

                                    {/* Generate Report Button */}
                                    <button
                                        onClick={handleGenerateReport}
                                        disabled={isGeneratingReport || isSimilarCaseMode}
                                        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg transition-all ${isGeneratingReport || isSimilarCaseMode
                                                ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30 cursor-not-allowed opacity-50'
                                                : 'bg-blue-500/20 text-blue-400 border border-blue-500/30 hover:bg-blue-500 hover:text-white active:scale-95 cursor-pointer'
                                            }`}
                                        title={isSimilarCaseMode ? "Không khả dụng khi so sánh ca bệnh" : "Sinh báo cáo"}
                                    >
                                        {isGeneratingReport ? (
                                            <>
                                                <div className="w-3.5 h-3.5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                                                <span className="font-medium">Đang tạo...</span>
                                            </>
                                        ) : (
                                            <>
                                                <FileText className="w-3.5 h-3.5" />
                                                <span className="font-medium">Sinh báo cáo</span>
                                            </>
                                        )}
                                    </button>

                                    {/* AI Analyze Button */}
                                    <button
                                        onClick={handleAIAnalyze}
                                        disabled={isSimilarCaseMode}
                                        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg transition-all ${isSimilarCaseMode
                                                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30 cursor-not-allowed opacity-50'
                                                : 'bg-amber-500 text-white shadow-lg shadow-amber-500/50 hover:bg-amber-600 active:scale-95 cursor-pointer'
                                            }`}
                                        title={isSimilarCaseMode ? "Không khả dụng khi so sánh ca bệnh" : "Phân tích AI"}
                                    >
                                        <Sparkles className="w-3.5 h-3.5" />
                                        <span className="font-medium">Phân tích</span>
                                    </button>
                                </div>
                            </div>

                            {/* Scrollable Content */}
                            <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
                                {(showChatbot || isSimilarCaseMode || isLoadingSimilarAnalysis) ? (
                                    <div className="flex flex-col h-full">
                                        {/* Chatbot Messages */}
                                        <div className="flex-1 overflow-y-auto space-y-3 mb-4">
                                            <div className="flex items-start gap-2">
                                                <div className="w-8 h-8 rounded-full bg-teal-500 flex items-center justify-center flex-shrink-0">
                                                    <Bot className="w-4 h-4 text-white" />
                                                </div>
                                                <div className="flex-1 bg-white/5 rounded-lg p-3">
                                                    {isLoadingSimilarAnalysis ? (
                                                        <div className="flex items-center gap-2 text-sm text-gray-300">
                                                            <div className="w-4 h-4 border-2 border-teal-500 border-t-transparent rounded-full animate-spin"></div>
                                                            <span>Đang phân tích ca bệnh tương tự...</span>
                                                        </div>
                                                    ) : isSimilarCaseMode && similarCaseData ? (
                                                        <div className="text-sm text-gray-300 space-y-3">
                                                            <p className="font-semibold text-teal-400">Phân tích so sánh ca bệnh</p>

                                                            {(() => {
                                                                // Get findings from both cases - auto-generate if not analyzed yet
                                                                const currentFindings = analysisData?.findings || generateAnalysisData(patient.diagnosis, patient.image)?.findings || [];
                                                                const similarFindings = generateAnalysisData(similarCaseData.diagnosis, similarCaseData.imageUrl)?.findings || [];

                                                                // Find common and different findings
                                                                const currentFindingTexts = currentFindings.map(f => f.text);
                                                                const similarFindingTexts = similarFindings.map(f => f.text);
                                                                const commonFindings = currentFindingTexts.filter(text => similarFindingTexts.includes(text));
                                                                const uniqueCurrentFindings = currentFindingTexts.filter(text => !similarFindingTexts.includes(text));
                                                                const uniqueSimilarFindings = similarFindingTexts.filter(text => !currentFindingTexts.includes(text));

                                                                return (
                                                                    <>
                                                                        {/* Current Patient Findings (clickable) */}
                                                                        <div className="space-y-1.5">
                                                                            <p className="font-medium text-white">Triệu chứng ca bệnh nhân ({patient?.name}):</p>
                                                                            <div className="flex flex-wrap gap-2">
                                                                                {currentFindings.map((finding, idx) => (
                                                                                    <button
                                                                                        key={idx}
                                                                                        onClick={(e) => {
                                                                                            e.preventDefault();
                                                                                            e.stopPropagation();
                                                                                            handleFindingClick(finding);
                                                                                        }}
                                                                                        className="px-2.5 py-1 bg-teal-500/20 hover:bg-teal-500/30 border border-teal-500/40 text-teal-300 rounded-md text-xs transition-colors cursor-pointer"
                                                                                    >
                                                                                        {finding.text}
                                                                                    </button>
                                                                                ))}
                                                                            </div>
                                                                        </div>

                                                                        {/* Similar Case Findings (display only) */}
                                                                        <div className="space-y-1.5">
                                                                            <p className="font-medium text-white">Triệu chứng ca tương tự ({similarCaseData.patientName}):</p>
                                                                            <div className="flex flex-wrap gap-2">
                                                                                {similarFindings.map((finding, idx) => (
                                                                                    <span
                                                                                        key={idx}
                                                                                        className="px-2.5 py-1 bg-gray-700/40 border border-gray-600/40 text-gray-300 rounded-md text-xs"
                                                                                    >
                                                                                        {finding.text}
                                                                                    </span>
                                                                                ))}
                                                                            </div>
                                                                        </div>

                                                                        {/* Comparison Analysis */}
                                                                        <div className="pt-2 border-t border-white/10 space-y-2.5">
                                                                            {commonFindings.length > 0 && (
                                                                                <p className="leading-relaxed">
                                                                                    <span className="font-medium text-green-400">Điểm tương đồng:</span> Cả hai ca bệnh đều có cùng triệu chứng <span className="text-green-300 font-medium">{commonFindings.join(', ')}</span>. Các dấu hiệu hình ảnh học thể hiện mức độ tổn thương phổi tương tự, với vị trí và phân bố tổn thương có nhiều điểm chung. Điều này gợi ý tiến trình bệnh lý và cơ chế gây tổn thương có sự tương đồng giữa hai ca.
                                                                                </p>
                                                                            )}

                                                                            {(uniqueCurrentFindings.length > 0 || uniqueSimilarFindings.length > 0) && (
                                                                                <p className="leading-relaxed">
                                                                                    <span className="font-medium text-amber-400">Điểm khác biệt:</span> Tuy nhiên, ca bệnh hiện tại
                                                                                    {uniqueCurrentFindings.length > 0 && (
                                                                                        <span> có thêm triệu chứng <span className="text-amber-300 font-medium">{uniqueCurrentFindings.join(', ')}</span></span>
                                                                                    )}
                                                                                    {uniqueCurrentFindings.length > 0 && uniqueSimilarFindings.length > 0 && ', trong khi'}
                                                                                    {uniqueSimilarFindings.length > 0 && (
                                                                                        <span> ca tương tự có <span className="text-gray-300 font-medium">{uniqueSimilarFindings.join(', ')}</span></span>
                                                                                    )}
                                                                                    . Sự khác biệt này cần được xem xét trong bối cảnh lâm sàng tổng thể, tiền sử bệnh lý nền và các yếu tố nguy cơ đặc thù của từng bệnh nhân.
                                                                                </p>
                                                                            )}

                                                                            <p className="leading-relaxed text-xs text-gray-400 italic mt-2">
                                                                                * Click vào triệu chứng màu xanh để xem hình ảnh xAI chi tiết
                                                                            </p>
                                                                        </div>
                                                                    </>
                                                                );
                                                            })()}
                                                        </div>
                                                    ) : (
                                                        <p className="text-sm text-gray-300">
                                                            Xin chào! Tôi là trợ lý AI y khoa. Tôi có thể giúp bạn hiểu rõ hơn về ca bệnh này, giải thích các phát hiện, hoặc trả lời câu hỏi của bạn về chẩn đoán và điều trị.
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Chatbot Input */}
                                        <div className="border-t border-white/10 pt-3">
                                            <div className="flex gap-2">
                                                <input
                                                    type="text"
                                                    placeholder="Đặt câu hỏi cho AI..."
                                                    className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-teal-500"
                                                />
                                                <button className="bg-teal-500 text-white rounded-lg px-4 py-2 hover:bg-teal-600 transition-colors">
                                                    <Send className="w-4 h-4" />
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                ) : isAnalyzing ? (
                                    <div className="flex items-center justify-center h-full text-center">
                                        <div>
                                            <div className="relative w-16 h-16 mx-auto mb-4">
                                                <div className="absolute inset-0 border-4 border-amber-500/20 rounded-full"></div>
                                                <div className="absolute inset-0 border-4 border-transparent border-t-amber-500 rounded-full animate-spin"></div>
                                                <Sparkles className="w-8 h-8 text-amber-500 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
                                            </div>
                                            <p className="text-base font-medium text-white mb-1">Đang suy luận...</p>
                                            <p className="text-sm text-gray-400">AI đang phân tích hình ảnh y tế</p>
                                        </div>
                                    </div>
                                ) : analysisData ? (
                                    <div className="space-y-4">
                                        <AnalysisTab
                                            findings={analysisData.findings}
                                            suspectedDiseases={analysisData.suspectedDiseases}
                                            onFindingClick={handleFindingClick}
                                            onFindingSelectionChange={handleFindingSelectionChange}
                                            onUpdateClick={handleUpdateClick}
                                            isUpdating={isUpdating}
                                            selectedFindingIds={selectedFindingIds}
                                            selectedFindingId={selectedFindingId}
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

            {/* Report Modal */}
            {showReportModal && reportData && (
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="bg-white border border-gray-200 rounded-xl max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col">
                        {/* Modal Header */}
                        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-gray-50">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center">
                                    <FileText className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <h3 className="text-lg font-semibold text-gray-800">Báo Cáo Chẩn Đoán Hình Ảnh</h3>
                                    <p className="text-sm text-gray-600">Bệnh nhân: {patient.name}</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setShowReportModal(false)}
                                className="text-gray-400 hover:text-gray-600 transition-colors"
                            >
                                <X className="w-6 h-6" />
                            </button>
                        </div>

                        {/* Modal Content - Scrollable PDF */}
                        <div className="flex-1 overflow-y-auto p-6">
                            <ReportPDF reportData={reportData} patient={patient} selectedImage={selectedImage} analysisData={analysisData} />
                        </div>
                    </div>
                </div>
            )}

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
