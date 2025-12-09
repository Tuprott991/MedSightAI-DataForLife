import { Eye, Calendar, User, Activity } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { getTranslatedDiagnosis } from '../../../utils/diagnosisHelper';

export const SimilarCaseCard = ({ caseData, onSelect, isSelected }) => {
    const { t } = useTranslation();
    const {
        id,
        patientName,
        diagnosis,
        imageUrl,
        similarity,
        status
    } = caseData;

    return (
        <div
            className={`bg-[#141414] border rounded-lg overflow-hidden transition-all ${isSelected
                ? 'border-teal-500 shadow-lg shadow-teal-500/20'
                : 'border-white/10'
                }`}
        >
            {/* Image Section */}
            <div className="relative aspect-video bg-black/50 overflow-hidden">
                <img
                    src={imageUrl}
                    alt={`Case ${id}`}
                    className="w-full h-full object-cover"
                />
                {/* Similarity Badge */}
                <div className="absolute top-2 right-2 bg-teal-500 text-white text-xs font-bold px-2 py-1 rounded">
                    {similarity}% {t('similarCase.match')}
                </div>
                {/* Status Badge */}
                {status && (
                    <div className="absolute top-2 left-2 bg-black/70 backdrop-blur-sm text-white text-xs px-2 py-1 rounded">
                        {status === 'Resolved' ? t('similarCase.resolved') :
                            status === 'Stable' ? t('similarCase.stable') :
                                status === 'Under Treatment' ? t('similarCase.underTreatment') :
                                    status === 'Critical' ? t('similarCase.critical') : status}
                    </div>
                )}
            </div>

            {/* Info Section */}
            <div className="p-3 space-y-2">
                {/* Patient Info */}
                <div>
                    <h3 className="text-sm font-semibold text-white mb-1 line-clamp-1">
                        {patientName}
                    </h3>
                    <p className="text-xs text-gray-400 line-clamp-2">
                        {getTranslatedDiagnosis(diagnosis, t)}
                    </p>
                </div>

                {/* View Details Button */}
                <button
                    onClick={() => onSelect(caseData)}
                    className={`w-full mt-2 flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs border rounded transition-all ${isSelected
                        ? 'bg-teal-500 border-teal-500 text-white'
                        : 'bg-white/5 hover:bg-white/10 border-white/10 hover:border-teal-500/50 text-gray-300 hover:text-white cursor-pointer'
                        }`}
                >
                    <Eye className="w-3 h-3" />
                    <span>{isSelected ? t('similarCase.selected') : t('similarCase.viewDetails')}</span>
                </button>
            </div>
        </div>
    );
};
