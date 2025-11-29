import { AlertCircle } from 'lucide-react';
import { Thermometer } from 'lucide-react';

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

export const KeyFindings = ({ findings, onFindingClick }) => {
    const sortedFindings = [...findings].sort((a, b) => b.confidence - a.confidence);

    const handleFindingClick = async (finding) => {
        if (onFindingClick) {
            // Mock API call - simulate fetching related images
            // TODO: Replace with actual API endpoint
            // const response = await fetch(`/api/findings/${finding.id}/images`);
            // const data = await response.json();

            // Mock: Return 2 images for demonstration
            const mockImagesData = [
                {
                    id: 998,
                    url: "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=800",
                    type: `Hình ảnh 1: ${finding.text}`,
                    imageCode: `REL-${finding.id}-1`,
                    modality: "AI-Enhanced"
                },
                {
                    id: 999,
                    url: "https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?w=800",
                    type: `Hình ảnh 2: ${finding.text}`,
                    imageCode: `REL-${finding.id}-2`,
                    modality: "AI-Enhanced"
                }
            ];

            onFindingClick(mockImagesData);
        }
    };

    return (
        <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-3">
                <Thermometer className="w-4 h-4 text-teal-500" />
                Các triệu chứng
            </h3>
            <div className="space-y-2">
                {sortedFindings.map((finding) => (
                    <div
                        key={finding.id}
                        onClick={() => handleFindingClick(finding)}
                        className={`p-2.5 rounded-lg border border-white ${getConfidenceBgColor(finding.confidence)} cursor-pointer hover:opacity-80 transition-opacity`}
                    >
                        <div className="flex items-start justify-between gap-3">
                            <p className="text-xs text-gray-300 flex-1">{finding.text}</p>
                            <span className={`text-xs font-bold ${getConfidenceColor(finding.confidence)} shrink-0`}>
                                {finding.confidence}%
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
