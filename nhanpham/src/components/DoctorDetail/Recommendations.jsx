import { TrendingUp, CheckCircle } from 'lucide-react';

export const Recommendations = ({ recommendations }) => {
    return (
        <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-3">
                <TrendingUp className="w-4 h-4 text-teal-500" />
                Recommendations
            </h3>
            <ul className="space-y-1.5">
                {recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start gap-2 text-xs text-gray-300 p-2 bg-[#0f0f0f] rounded border border-white/5">
                        <CheckCircle className="w-3.5 h-3.5 text-teal-500 mt-0.5 shrink-0" />
                        <span>{rec}</span>
                    </li>
                ))}
            </ul>
        </div>
    );
};
