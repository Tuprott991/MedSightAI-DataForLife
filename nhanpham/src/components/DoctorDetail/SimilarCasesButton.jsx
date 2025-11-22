import { Search } from 'lucide-react';

export const SimilarCasesButton = ({ onClick }) => {
    return (
        <button
            onClick={onClick}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-500/50 text-gray-300 hover:text-white rounded transition-all font-medium"
            title="Find similar medical cases"
        >
            <Search className="w-3.5 h-3.5" />
            <span>Similar Cases</span>
        </button>
    );
};
