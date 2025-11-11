import { AlertCircle } from 'lucide-react';

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

export const KeyFindings = ({ findings }) => {
    return (
        <div>
            <h3 className="text-sm font-semibold text-white mb-3">Key Findings</h3>
            <div className="space-y-2">
                {findings.map((finding) => (
                    <div key={finding.id} className="p-2.5 bg-[#0f0f0f] rounded-lg border border-white/5 flex items-start gap-2">
                        <AlertCircle className={`w-4 h-4 mt-0.5 shrink-0 ${getSeverityColor(finding.severity)}`} />
                        <div className="flex-1 min-w-0">
                            <p className="text-xs text-gray-300 leading-relaxed">{finding.text}</p>
                            <p className="text-xs text-gray-600 mt-1">Confidence: {finding.confidence}%</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
