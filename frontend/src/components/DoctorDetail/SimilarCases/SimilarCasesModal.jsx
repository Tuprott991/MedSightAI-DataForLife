import { X, Loader2, AlertCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { generateSimilarityCam, searchSimilarCases } from '../../../services/patientApi';
import { getTranslatedDiagnosis } from '../../../utils/diagnosisHelper';
import { SimilarCaseCard } from './SimilarCaseCard';

const DEFAULT_TOP_K = 6;
const GENERATING_SALIENCY_LABEL = 'Generating saliency map...';

const translateStatus = (status, t) => {
    const statusMap = {
        Resolved: t('similarCase.resolved'),
        Stable: t('similarCase.stable'),
        'Under Treatment': t('similarCase.underTreatment'),
        Critical: t('similarCase.critical'),
        Processed: 'Processed',
        Unprocessed: 'Unprocessed',
        Unprocesed: 'Unprocessed',
    };

    return statusMap[status] || status || '-';
};

const translateGender = (gender, language) => {
    if (gender === 'M') return language === 'vi' ? 'Nam' : 'Male';
    if (gender === 'F') return language === 'vi' ? 'Nữ' : 'Female';
    return gender || '-';
};

const normalizeSimilarCases = (response, t) =>
    (response.case_details || []).map((caseDetail, index) => {
        const rawScore = response.similarity_scores?.[index] ?? 0;
        const similarityPercent = Number((rawScore * 100).toFixed(1));
        return {
            id: caseDetail.case_id,
            caseId: caseDetail.case_id,
            patientId: caseDetail.patient_id,
            patientName: caseDetail.patient_name || `Case ${index + 1}`,
            age: caseDetail.age ?? '-',
            gender: caseDetail.gender || '-',
            diagnosis: caseDetail.diagnosis || t('similarCase.noResults'),
            imageUrl: caseDetail.image_path || caseDetail.processed_img_path || '',
            similarity: Math.max(0, Math.min(100, similarityPercent)),
            similarityScore: rawScore,
            date: caseDetail.timestamp,
            status: caseDetail.status || 'Processed',
        };
    });

export const SimilarCasesModal = ({ isOpen, onClose, currentImage, patientInfo, onCompareImages }) => {
    const { t, i18n } = useTranslation();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [similarCases, setSimilarCases] = useState([]);
    const [selectedCase, setSelectedCase] = useState(null);
    const [requestVersion, setRequestVersion] = useState(0);
    const [comparingCaseId, setComparingCaseId] = useState(null);

    const currentCaseId = patientInfo?.latest_case?.id || null;
    const currentImagePath = patientInfo?.latest_case?.image_path || currentImage?.url || null;
    const originalCaseImageUrl = patientInfo?.latest_case?.image_path || currentImage?.url || null;

    useEffect(() => {
        if (isOpen) {
            setSelectedCase(null);
        }
    }, [isOpen]);

    useEffect(() => {
        let isMounted = true;

        const fetchSimilarCases = async () => {
            if (!isOpen || (!currentCaseId && !currentImagePath)) {
                return;
            }

            setLoading(true);
            setError(null);

            try {
                const response = await searchSimilarCases({
                    caseId: currentCaseId,
                    imagePath: currentCaseId ? null : currentImagePath,
                    topK: DEFAULT_TOP_K,
                });

                if (!isMounted) {
                    return;
                }

                const normalizedCases = normalizeSimilarCases(response, t);
                setSimilarCases(normalizedCases);
                setSelectedCase(normalizedCases[0] || null);
            } catch (err) {
                if (!isMounted) {
                    return;
                }

                setError(err.message || t('similarCase.errorLoading'));
                setSimilarCases([]);
                setSelectedCase(null);
                console.error('Error fetching similar cases:', err);
            } finally {
                if (isMounted) {
                    setLoading(false);
                }
            }
        };

        fetchSimilarCases();

        return () => {
            isMounted = false;
        };
    }, [currentCaseId, currentImagePath, isOpen, requestVersion, t]);

    useEffect(() => {
        const handleEscape = (event) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };

        if (isOpen) {
            document.addEventListener('keydown', handleEscape);
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
            <div
                className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 transition-opacity"
                onClick={onClose}
            />

            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
                <div
                    className="bg-[#1a1a1a] border border-white/10 rounded-2xl shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col pointer-events-auto"
                    onClick={(event) => event.stopPropagation()}
                >
                    <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-[#141414] rounded-t-2xl shrink-0">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-teal-500/20 rounded-lg flex items-center justify-center">
                                <span className="text-teal-500 text-lg font-bold">SC</span>
                            </div>
                            <div>
                                <h2 className="text-xl font-bold text-white">{t('similarCase.title')}</h2>
                                <p className="text-xs text-gray-400">{t('similarCase.aiAnalysisResult')}</p>
                            </div>
                        </div>

                        <button
                            onClick={onClose}
                            className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                            title={t('similarCase.close')}
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    <div className="flex-1 overflow-hidden flex gap-4 p-6">
                        <div className="flex-4 flex flex-col">
                            <div className="mb-4">
                                <h3 className="text-sm font-semibold text-white mb-1">
                                    {loading ? t('similarCase.searching') : `${t('similarCase.found')} ${similarCases.length} ${t('similarCase.title')}`}
                                </h3>
                                <p className="text-xs text-gray-400">
                                    {t('similarCase.basedOn')}
                                </p>
                            </div>

                            <div className="flex-1 overflow-y-auto custom-scrollbar">
                                {loading ? (
                                    <div className="flex items-center justify-center h-full">
                                        <div className="text-center">
                                            <Loader2 className="w-12 h-12 text-teal-500 mx-auto mb-3 animate-spin" />
                                            <p className="text-sm text-gray-400">{t('similarCase.analyzingCases')}</p>
                                        </div>
                                    </div>
                                ) : error ? (
                                    <div className="flex items-center justify-center h-full">
                                        <div className="text-center">
                                            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-3" />
                                            <p className="text-sm text-gray-400 mb-3">{error}</p>
                                            <button
                                                onClick={() => {
                                                    setError(null);
                                                    setSelectedCase(null);
                                                    setSimilarCases([]);
                                                    setRequestVersion((previous) => previous + 1);
                                                }}
                                                className="px-4 py-2 text-sm bg-teal-500 hover:bg-teal-600 text-white rounded-lg transition-colors"
                                            >
                                                {t('similarCase.retry')}
                                            </button>
                                        </div>
                                    </div>
                                ) : similarCases.length > 0 ? (
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
                                    <div className="flex items-center justify-center h-full">
                                        <div className="text-center">
                                            <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
                                                <span className="text-2xl">🔍</span>
                                            </div>
                                            <p className="text-sm text-gray-400">{t('similarCase.noResults')}</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="flex-1 bg-[#141414] border border-white/10 rounded-lg p-4">
                            {selectedCase ? (
                                <div className="space-y-4">
                                    <div>
                                        <h3 className="text-sm font-semibold text-white mb-2">{t('similarCase.caseDetails')}</h3>
                                        <div className="space-y-2">
                                            <div>
                                                <p className="text-xs text-gray-500">{t('similarCase.patient')}</p>
                                                <p className="text-sm text-white">{selectedCase.patientName}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">{t('similarCase.ageGender')}</p>
                                                <p className="text-sm text-white">
                                                    {selectedCase.age} {t('similarCase.yearsOld')}, {translateGender(selectedCase.gender, i18n.language)}
                                                </p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">{t('similarCase.diagnosis')}</p>
                                                <p className="text-sm text-white">{getTranslatedDiagnosis(selectedCase.diagnosis, t)}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">{t('similarCase.examinationDate')}</p>
                                                <p className="text-sm text-white">
                                                    {selectedCase.date
                                                        ? new Date(selectedCase.date).toLocaleDateString(i18n.language === 'vi' ? 'vi-VN' : 'en-US')
                                                        : '-'}
                                                </p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">{t('similarCase.status')}</p>
                                                <p className="text-sm text-white">{translateStatus(selectedCase.status, t)}</p>
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500">{t('similarCase.similarity')}</p>
                                                <div className="flex items-center gap-2">
                                                    <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                                                        <div
                                                            className="h-full bg-teal-500 rounded-full transition-all"
                                                            style={{ width: `${selectedCase.similarity}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-sm text-teal-500 font-semibold">
                                                        {selectedCase.similarity.toFixed(1)}%
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="pt-4 border-t border-white/10">
                                        <button
                                            onClick={async () => {
                                                if (!selectedCase.imageUrl || !currentCaseId) {
                                                    return;
                                                }

                                                setComparingCaseId(selectedCase.id);
                                                setError(null);

                                                try {
                                                    const camResult = await generateSimilarityCam({
                                                        caseId: currentCaseId,
                                                        similarCaseId: selectedCase.id,
                                                    });

                                                    const queryOverlayUrl = camResult.query_overlay_b64
                                                        ? `data:image/png;base64,${camResult.query_overlay_b64}`
                                                        : originalCaseImageUrl;
                                                    const similarOverlayUrl = camResult.similar_overlay_b64
                                                        ? `data:image/png;base64,${camResult.similar_overlay_b64}`
                                                        : selectedCase.imageUrl;

                                                    const comparisonImages = [
                                                        {
                                                            id: `query-${currentCaseId}`,
                                                            url: originalCaseImageUrl || queryOverlayUrl,
                                                            type: t('imageViewer.patientImage'),
                                                            imageCode: `CASE-${String(currentCaseId).slice(0, 8)}`,
                                                            modality: 'Comparison',
                                                        },
                                                        {
                                                            id: `similar-${selectedCase.id}`,
                                                            url: similarOverlayUrl,
                                                            type: `${t('similarCase.similarCase')}: ${selectedCase.patientName}`,
                                                            imageCode: `SIMILAR-${String(selectedCase.id).slice(0, 8)}`,
                                                            modality: 'Saliency',
                                                        },
                                                    ];

                                                    onCompareImages(comparisonImages, {
                                                        id: selectedCase.id,
                                                        patientId: selectedCase.patientId,
                                                        patientName: selectedCase.patientName,
                                                        diagnosis: selectedCase.diagnosis,
                                                        imageUrl: similarOverlayUrl,
                                                        originalImageUrl: selectedCase.imageUrl,
                                                        queryCamImageUrl: queryOverlayUrl,
                                                        similarity: selectedCase.similarity,
                                                    });
                                                    onClose();
                                                } catch (err) {
                                                    setError(err.message || t('similarCase.errorLoading'));
                                                    console.error('Error generating similarity CAM:', err);
                                                } finally {
                                                    setComparingCaseId(null);
                                                }
                                            }}
                                            disabled={!selectedCase.imageUrl || !currentCaseId || comparingCaseId === selectedCase.id}
                                            className="w-full px-3 py-2 text-xs bg-teal-500 hover:bg-teal-600 disabled:bg-white/10 disabled:text-gray-500 text-white rounded-lg transition-colors font-medium flex items-center justify-center gap-2"
                                        >
                                            {comparingCaseId === selectedCase.id ? (
                                                <>
                                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                    <span>{GENERATING_SALIENCY_LABEL}</span>
                                                </>
                                            ) : (
                                                <span>{t('similarCase.compareImages')}</span>
                                            )}
                                            
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex items-center justify-center h-full text-center">
                                    <p className="text-xs text-gray-500">
                                        {t('similarCase.selectCase')}
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

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
