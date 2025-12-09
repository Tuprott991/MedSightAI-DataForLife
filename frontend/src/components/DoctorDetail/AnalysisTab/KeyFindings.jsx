import { AlertCircle, RefreshCw } from 'lucide-react';
import { Thermometer } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

const getSeverityColor = (severity) => {
    switch (severity) {
        case 'high':
            return 'text-red-400';
        case 'medium':
            return 'text-yellow-400';
        case 'low':
            return 'text-teal-400';
        default:
            return 'text-gray-400';
    }
};

const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'text-red-400';
    if (confidence >= 80) return 'text-orange-400';
    if (confidence >= 70) return 'text-yellow-400';
    if (confidence >= 60) return 'text-lime-400';
    return 'text-emerald-400';
};

const getConfidenceBarColor = (confidence) => {
    if (confidence >= 90) return 'bg-red-500';
    if (confidence >= 80) return 'bg-orange-500';
    if (confidence >= 70) return 'bg-yellow-500';
    if (confidence >= 60) return 'bg-lime-500';
    return 'bg-emerald-500';
};

const getConfidenceBgColor = (confidence) => {
    if (confidence >= 90) return 'bg-red-500/40';
    if (confidence >= 80) return 'bg-orange-500/40';
    if (confidence >= 70) return 'bg-yellow-500/40';
    if (confidence >= 60) return 'bg-lime-500/40';
    return 'bg-emerald-500/40';
};

export const KeyFindings = ({ findings, onFindingClick, onFindingSelectionChange, selectedFindingId }) => {
    const { t } = useTranslation();
    const [selectedFindings, setSelectedFindings] = useState(() =>
        findings.reduce((acc, f) => ({ ...acc, [f.id]: true }), {})
    );

    const sortedFindings = [...findings].sort((a, b) => b.confidence - a.confidence);

    const handleCheckboxChange = (findingId) => {
        setSelectedFindings(prev => ({
            ...prev,
            [findingId]: !prev[findingId]
        }));
    };

    const handleUpdate = async () => {
        setIsUpdating(true);

        // Mô phỏng API call
        const selectedIds = Object.entries(selectedFindings)
            .filter(([_, isSelected]) => isSelected)
            .map(([id, _]) => parseInt(id));

        // Mock: Simulate API call với delay
        setTimeout(() => {
            if (onUpdateDiagnosis) {
                onUpdateDiagnosis(selectedIds);
            }
            setIsUpdating(false);
        }, 1500);
    };

    const handleFindingClick = async (finding) => {
        if (onFindingClick) {
            // Pass the finding object to parent
            onFindingClick(finding);
        }
    };

    return (
        <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-3">
                <Thermometer className="w-4 h-4 text-teal-500" />
                {t('doctorDetail.symptoms')}
            </h3>
            <style>{`
                .checkbox-custom {
                    appearance: none;
                    -webkit-appearance: none;
                    background-color: #374151;
                    border: 1px solid #4b5563;
                }
                .checkbox-custom:checked {
                    background-color: #14b8a6;
                    border-color: #14b8a6;
                    background-image: url("data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3e%3c/svg%3e");
                }
            `}</style>
            <div className="space-y-2">
                {sortedFindings.map((finding) => (
                    <div
                        key={finding.id}
                        className={`p-2.5 rounded-lg border transition-all ${selectedFindingId === finding.id
                                ? 'border-teal-400 border-2 bg-teal-500/20 shadow-lg shadow-teal-500/20'
                                : `border-white ${getConfidenceBgColor(finding.confidence)}`
                            }`}
                    >
                        <div className="flex items-start gap-3">
                            <input
                                type="checkbox"
                                checked={selectedFindings[finding.id] || false}
                                onChange={() => handleCheckboxChange(finding.id)}
                                className="mt-0.5 w-4 h-4 rounded border-gray-600 cursor-pointer checkbox-custom"
                                onClick={(e) => e.stopPropagation()}
                            />
                            <div
                                className="flex-1 cursor-pointer hover:opacity-80"
                                onClick={() => handleFindingClick(finding)}
                            >
                                <div className="flex items-start justify-between gap-3">
                                    <p className="text-xs text-gray-300 flex-1">{finding.text}</p>
                                    <span className={`text-xs font-bold ${getConfidenceColor(finding.confidence)} shrink-0`}>
                                        {finding.confidence}%
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
