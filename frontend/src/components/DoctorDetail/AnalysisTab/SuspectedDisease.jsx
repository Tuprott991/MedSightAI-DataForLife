import { Activity } from 'lucide-react';

const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return 'text-emerald-400';
    if (confidence >= 60) return 'text-yellow-400';
    return 'text-orange-400';
};

const getConfidenceBarColor = (confidence) => {
    if (confidence >= 80) return 'bg-emerald-500';
    if (confidence >= 60) return 'bg-yellow-500';
    return 'bg-orange-500';
};

export const SuspectedDisease = ({ diseases }) => {
    const sortedDiseases = [...diseases].sort((a, b) => b.confidence - a.confidence);

    return (
        <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-3">
                <Activity className="w-4 h-4 text-teal-500" />
                Bệnh nghi ngờ
            </h3>
            <div className="space-y-2">
                {sortedDiseases.slice(0, 3).map((disease, index) => (
                    <div
                        key={index}
                        className="p-2.5 bg-[#0f0f0f] rounded-lg border border-white/5"
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="text-xs font-semibold text-gray-300">
                                    #{index + 1}
                                </span>
                                <p className="text-xs text-gray-300 font-medium">
                                    {disease.name}
                                </p>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
