import { X, Loader2, AlertCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
import { SimilarCaseCard } from './SimilarCaseCard';

export const SimilarCasesModal = ({ isOpen, onClose, currentImage, patientInfo, onCompareImages }) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [similarCases, setSimilarCases] = useState([]);
    const [selectedCase, setSelectedCase] = useState(null);

    // Reset selected case when modal opens
    useEffect(() => {
        if (isOpen) {
            setSelectedCase(null); // Reset selection when modal opens
        }
    }, [isOpen]);

    // Fetch similar cases when modal opens
    useEffect(() => {
        const fetchSimilarCases = async () => {
            if (!isOpen || !currentImage) return;

            setLoading(true);
            setError(null);

            try {
                // TODO: Replace with actual API endpoint
                // const response = await fetch(`/api/similar-cases`, {
                //     method: 'POST',
                //     headers: { 'Content-Type': 'application/json' },
                //     body: JSON.stringify({
                //         imageUrl: currentImage.url,
                //         patientId: patientInfo?.id,
                //         diagnosis: patientInfo?.diagnosis
                //     })
                // });
                // const data = await response.json();

                // Mock API call - simulate network delay
                await new Promise(resolve => setTimeout(resolve, 1500));

                // Determine disease category based on current patient diagnosis
                const currentDiagnosis = patientInfo?.diagnosis || "";
                let diseaseCategory = "";

                if (currentDiagnosis.includes("Viêm phổi")) {
                    diseaseCategory = "Viêm phổi";
                } else if (currentDiagnosis.includes("Lao phổi")) {
                    diseaseCategory = "Lao phổi";
                } else if (currentDiagnosis.includes("COVID-19")) {
                    diseaseCategory = "COVID-19";
                } else {
                    diseaseCategory = currentDiagnosis; // fallback
                }

                // Mock data with Vietnamese names and related diagnoses
                const mockData = [
                    {
                        id: 1,
                        patientName: "Nguyễn Văn An",
                        age: 52,
                        gender: "Nam",
                        diagnosis: diseaseCategory,
                        imageUrl: "/src/mock_data/patient_data/01_Tuberculosis/origin.png",
                        similarity: 94,
                        date: "2025-10-15",
                        status: "Đã khỏi"
                    },
                    {
                        id: 2,
                        patientName: "Trần Thị Bình",
                        age: 48,
                        gender: "Nữ",
                        diagnosis: diseaseCategory,
                        imageUrl: "/src/mock_data/patient_data/02_pneumonia/origin.png",
                        similarity: 89,
                        date: "2025-09-22",
                        status: "Ổn định"
                    },
                    {
                        id: 3,
                        patientName: "Lê Văn Cường",
                        age: 55,
                        gender: "Nam",
                        diagnosis: diseaseCategory,
                        imageUrl: "/src/mock_data/patient_data/03_Otherdisease/origin.png",
                        similarity: 87,
                        date: "2025-08-10",
                        status: "Đang điều trị"
                    },
                    {
                        id: 4,
                        patientName: "Phạm Thị Dung",
                        age: 46,
                        gender: "Nữ",
                        diagnosis: diseaseCategory,
                        imageUrl: "/src/mock_data/patient_data/04_Turbeculosis/origin.png",
                        similarity: 85,
                        date: "2025-11-01",
                        status: "Nguy kịch"
                    },
                    {
                        id: 5,
                        patientName: "Hoàng Văn Em",
                        age: 60,
                        gender: "Nam",
                        diagnosis: diseaseCategory,
                        imageUrl: "/src/mock_data/patient_data/05_pneumonia/origin.png",
                        similarity: 83,
                        date: "2025-07-18",
                        status: "Đã khỏi"
                    },
                    {
                        id: 6,
                        patientName: "Đỗ Thị Phương",
                        age: 51,
                        gender: "Nữ",
                        diagnosis: diseaseCategory,
                        imageUrl: "/src/mock_data/patient_data/06_pneumonia/origin.png",
                        similarity: 81,
                        date: "2025-06-25",
                        status: "Ổn định"
                    }
                ];

                setSimilarCases(mockData);
                setLoading(false);
            } catch (err) {
                setError('Không thể tải các ca bệnh tương tự. Vui lòng thử lại.');
                setLoading(false);
                console.error('Error fetching similar cases:', err);
            }
        };

        fetchSimilarCases();
    }, [isOpen, currentImage, patientInfo]);

    // Close modal on ESC key
    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };

        if (isOpen) {
            document.addEventListener('keydown', handleEscape);
            // Prevent body scroll when modal is open
            document.body.style.overflow = 'hidden';
        }

        return () => {
            document.removeEventListener('keydown', handleEscape);
            document.body.style.overflow = 'unset';
        };
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <>
            {/* Backdrop */}
            <div
                className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 transition-opacity"
                onClick={onClose}
            />

            {/* Modal Container */}
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
                <div
                    className="bg-[#1a1a1a] border border-white/10 rounded-2xl shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col pointer-events-auto"
                    onClick={(e) => e.stopPropagation()}
                >
                    {/* Modal Header */}
                    <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-[#141414] rounded-t-2xl shrink-0">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-teal-500/20 rounded-lg flex items-center justify-center">
                                <span className="text-teal-500 text-lg font-bold">SC</span>
                            </div>
                            <div>
                                <h2 className="text-xl font-bold text-white">Ca Bệnh Tương Tự</h2>
                                <p className="text-xs text-gray-400">Kết quả phân tích AI dựa trên mẫu hình ảnh</p>
                            </div>
                        </div>

                        {/* Close Button */}
                        <button
                            onClick={onClose}
                            className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                            title="Đóng (ESC)"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Modal Content - Two Column Layout (4:1) */}
                    <div className="flex-1 overflow-hidden flex gap-4 p-6">
                        {/* Left Section - Similar Cases Grid (4/5) */}
                        <div className="flex-4 flex flex-col">
                            <div className="mb-4">
                                <h3 className="text-sm font-semibold text-white mb-1">
                                    {loading ? 'Đang tìm kiếm...' : `Tìm thấy ${similarCases.length} Ca Bệnh Tương Tự`}
                                </h3>
                                <p className="text-xs text-gray-400">
                                    Dựa trên mẫu hình ảnh và chẩn đoán
                                </p>
                            </div>

                            {/* Scrollable Cases Grid */}
                            <div className="flex-1 overflow-y-auto custom-scrollbar">
                                {loading ? (
                                    // Loading State
                                    <div className="flex items-center justify-center h-full">
                                        <div className="text-center">
                                            <Loader2 className="w-12 h-12 text-teal-500 mx-auto mb-3 animate-spin" />
                                            <p className="text-sm text-gray-400">Đang phân tích các ca bệnh tương tự...</p>
                                        </div>
                                    </div>
                                ) : error ? (
                                    // Error State
                                    <div className="flex items-center justify-center h-full">
                                        <div className="text-center">
                                            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-3" />
                                            <p className="text-sm text-gray-400 mb-3">{error}</p>
                                            <button
                                                onClick={() => window.location.reload()}
                                                className="px-4 py-2 text-sm bg-teal-500 hover:bg-teal-600 text-white rounded-lg transition-colors"
                                            >
                                                Thử Lại
                                            </button>
                                        </div>
                                    </div>
                                ) : similarCases.length > 0 ? (
                                    // Cases Grid
                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
                                        {similarCases.map((caseData) => (
                                            <SimilarCaseCard
                                                key={caseData.id}
                                                caseData={caseData}
                                                onSelect={setSelectedCase}
                                                isSelected={selectedCase?.id === caseData.id}
                                            />
                                        ))}
                                    </div>
                                ) : (
                                    // No Results
                                    <div className="flex items-center justify-center h-full">
                                        <div className="text-center">
                                            <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
                                                <span className="text-2xl">🔍</span>
                                            </div>
                                            <p className="text-sm text-gray-400">Không tìm thấy ca bệnh tương tự</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Right Section - Case Details (1/5) */}
                        <div className="flex-1 bg-[#141414] border border-white/10 rounded-lg p-4">
                            {selectedCase ? (
                                // Selected Case Details
                                <div className="space-y-4">
                                    <div>
                                        <h3 className="text-sm font-semibold text-white mb-2">Chi Tiết Ca Bệnh</h3>
                                        <div className="space-y-2">
                                            <div>
                                                <p className="text-xs text-gray-500">Bệnh Nhân</p>
                                                <p className="text-sm text-white">{selectedCase.patientName}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">Tuổi / Giới Tính</p>
                                                <p className="text-sm text-white">{selectedCase.age} tuổi, {selectedCase.gender === 'M' ? 'Nam' : 'Nữ'}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">Chẩn Đoán</p>
                                                <p className="text-sm text-white">{selectedCase.diagnosis}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">Ngày Khám</p>
                                                <p className="text-sm text-white">{new Date(selectedCase.date).toLocaleDateString('vi-VN')}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">Tình Trạng</p>
                                                <p className="text-sm text-white">{selectedCase.status === 'Resolved' ? 'Đã Hồi Phục' : selectedCase.status === 'Stable' ? 'Ổn Định' : selectedCase.status === 'Under Treatment' ? 'Đang Điều Trị' : selectedCase.status === 'Critical' ? 'Nguy Kịch' : selectedCase.status}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">Độ Tương Đồng</p>
                                                <div className="flex items-center gap-2">
                                                    <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                                                        <div
                                                            className="h-full bg-teal-500 rounded-full transition-all"
                                                            style={{ width: `${selectedCase.similarity}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-sm text-teal-500 font-semibold">
                                                        {selectedCase.similarity}%
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Action Buttons */}
                                    <div className="pt-4 border-t border-white/10">
                                        <button
                                            onClick={() => {
                                                // Create comparison image data
                                                const comparisonImages = [
                                                    currentImage, // Original image on the left
                                                    {
                                                        id: selectedCase.id + 1000,
                                                        url: selectedCase.imageUrl,
                                                        type: `Ca tương tự: ${selectedCase.patientName}`,
                                                        imageCode: `SIMILAR-${selectedCase.id}`,
                                                        modality: "Comparison"
                                                    }
                                                ];
                                                onCompareImages(comparisonImages);
                                                onClose();
                                            }}
                                            className="w-full px-3 py-2 text-xs bg-teal-500 hover:bg-teal-600 text-white rounded-lg transition-colors font-medium"
                                        >
                                            So Sánh Hình Ảnh
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                // No Selection Placeholder
                                <div className="flex items-center justify-center h-full text-center">
                                    <p className="text-xs text-gray-500">
                                        Chọn một ca bệnh để xem chi tiết
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Custom Scrollbar Styles */}
            <style jsx>{`
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
        </>
    );
};
