import { Eye, Calendar, User, Activity } from 'lucide-react';

export const SimilarCaseCard = ({ caseData, onSelect, isSelected }) => {
    const {
        id,
        patientName,
        age,
        gender,
        diagnosis,
        imageUrl,
        similarity,
        date,
        status
    } = caseData;

    return (
        <div
            onClick={() => onSelect(caseData)}
            className={`bg-[#141414] border rounded-lg overflow-hidden cursor-pointer transition-all hover:scale-[1.02] ${isSelected
                    ? 'border-teal-500 shadow-lg shadow-teal-500/20'
                    : 'border-white/10 hover:border-white/20'
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
                    {similarity}% Match
                </div>
                {/* Status Badge */}
                {status && (
                    <div className="absolute top-2 left-2 bg-black/70 backdrop-blur-sm text-white text-xs px-2 py-1 rounded">
                        {status}
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
                        {diagnosis}
                    </p>
                </div>

                {/* Details Grid */}
                <div className="grid grid-cols-2 gap-2 pt-2 border-t border-white/10">
                    <div className="flex items-center gap-1.5">
                        <User className="w-3 h-3 text-gray-500" />
                        <span className="text-xs text-gray-400">
                            {age}y, {gender}
                        </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <Calendar className="w-3 h-3 text-gray-500" />
                        <span className="text-xs text-gray-400">{date}</span>
                    </div>
                </div>

                {/* View Details Link */}
                <button
                    className="w-full mt-2 flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-500/50 text-gray-300 hover:text-white rounded transition-all"
                >
                    <Eye className="w-3 h-3" />
                    <span>View Details</span>
                </button>
            </div>
        </div>
    );
};
