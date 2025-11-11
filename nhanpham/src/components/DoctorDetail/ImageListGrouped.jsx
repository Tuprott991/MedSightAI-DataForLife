import { useState } from 'react';
import { ChevronDown, ChevronRight, Calendar, Folder, MoreVertical, User } from 'lucide-react';

export const ImageListGrouped = ({ imageGroups, selectedImage, onImageSelect, patient }) => {
    const [expandedGroups, setExpandedGroups] = useState([imageGroups[0]?.id]);
    const [viewMode, setViewMode] = useState('images'); // 'images' or 'patient'
    const [dropdownOpen, setDropdownOpen] = useState(false);

    const toggleGroup = (groupId) => {
        setExpandedGroups(prev =>
            prev.includes(groupId)
                ? prev.filter(id => id !== groupId)
                : [...prev, groupId]
        );
    };

    const isExpanded = (groupId) => expandedGroups.includes(groupId);

    const getTitle = () => {
        return viewMode === 'images' ? 'Medical Images' : 'Patient Information';
    };

    const getIcon = () => {
        return viewMode === 'images'
            ? <Folder className="w-4 h-4 text-teal-500" />
            : <User className="w-4 h-4 text-teal-500" />;
    };

    return (
        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden flex flex-col h-[calc(100vh-110px)]">
            {/* Header with Dropdown */}
            <div className="px-4 py-3 border-b border-white/10 bg-[#141414] flex items-center justify-between">
                <h3 className="text-base font-semibold text-white flex items-center gap-2">
                    {getIcon()}
                    {getTitle()}
                </h3>

                {/* Dropdown Menu */}
                <div className="relative">
                    <button
                        onClick={() => setDropdownOpen(!dropdownOpen)}
                        className="p-1 hover:bg-white/10 rounded transition-colors"
                    >
                        <MoreVertical className="w-4 h-4 text-gray-400" />
                    </button>

                    {dropdownOpen && (
                        <div className="absolute right-0 mt-2 w-48 bg-[#1a1a1a] border border-white/10 rounded-lg shadow-lg z-10">
                            <button
                                onClick={() => {
                                    setViewMode('images');
                                    setDropdownOpen(false);
                                }}
                                className={`w-full text-left px-4 py-2 text-sm transition-colors flex items-center gap-2 ${viewMode === 'images'
                                        ? 'bg-teal-500/20 text-teal-400'
                                        : 'text-gray-300 hover:bg-white/5'
                                    }`}
                            >
                                <Folder className="w-4 h-4" />
                                Medical Images
                            </button>
                            <button
                                onClick={() => {
                                    setViewMode('patient');
                                    setDropdownOpen(false);
                                }}
                                className={`w-full text-left px-4 py-2 text-sm transition-colors rounded-b-lg flex items-center gap-2 ${viewMode === 'patient'
                                        ? 'bg-teal-500/20 text-teal-400'
                                        : 'text-gray-300 hover:bg-white/5'
                                    }`}
                            >
                                <User className="w-4 h-4" />
                                Patient Information
                            </button>
                        </div>
                    )}
                </div>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar p-3">
                {viewMode === 'images' ? (
                    /* Medical Images View */
                    <div className="space-y-2">
                        {imageGroups.map((group) => (
                            <div key={group.id} className="bg-[#0f0f0f] border border-white/5 rounded-lg overflow-hidden">
                                {/* Group Header */}
                                <button
                                    onClick={() => toggleGroup(group.id)}
                                    className="w-full px-3 py-2.5 flex items-center justify-between hover:bg-white/5 transition-colors"
                                >
                                    <div className="flex items-center gap-2">
                                        {isExpanded(group.id) ? (
                                            <ChevronDown className="w-4 h-4 text-teal-500" />
                                        ) : (
                                            <ChevronRight className="w-4 h-4 text-gray-500" />
                                        )}
                                        <div className="text-left">
                                            <p className="text-xs font-medium text-white">{group.examType}</p>
                                            <p className="text-xs text-gray-500 flex items-center gap-1">
                                                <Calendar className="w-3 h-3" />
                                                {new Date(group.examDate).toLocaleDateString()}
                                            </p>
                                        </div>
                                    </div>
                                    <span className={`text-xs px-2 py-0.5 rounded-full border ${isExpanded(group.id)
                                        ? 'bg-teal-500/20 text-teal-400 border-teal-500/30'
                                        : 'bg-white/5 text-gray-400 border-white/10'
                                        }`}>
                                        {group.images.length}
                                    </span>
                                </button>

                                {/* Group Images */}
                                {isExpanded(group.id) && (
                                    <div className="p-2 space-y-1.5 border-t border-white/5">
                                        {group.images.map((image) => (
                                            <div
                                                key={image.id}
                                                onClick={() => onImageSelect(image)}
                                                className={`cursor-pointer rounded-md overflow-hidden border transition-all ${selectedImage?.id === image.id
                                                    ? 'border-teal-500 bg-teal-500/10'
                                                    : 'border-white/5 hover:border-teal-500/50 hover:bg-white/5'
                                                    }`}
                                            >
                                                <div className="flex gap-2 p-2">
                                                    <div className="w-16 h-16 shrink-0 rounded overflow-hidden bg-black/30">
                                                        <img
                                                            src={image.url}
                                                            alt={image.type}
                                                            className="w-full h-full object-cover"
                                                        />
                                                    </div>
                                                    <div className="flex-1 min-w-0">
                                                        <p className="text-xs font-medium text-white truncate">{image.type}</p>
                                                        <p className="text-xs text-teal-400 mt-0.5">{image.imageCode}</p>
                                                        <span className="inline-block text-xs px-1.5 py-0.5 bg-white/5 text-gray-400 rounded mt-1">
                                                            {image.modality}
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                ) : (
                    /* Patient Information View */
                    <div className="space-y-3">
                        <div className="bg-[#0f0f0f] border border-white/5 rounded-lg p-3">
                            <h4 className="text-xs font-semibold text-teal-400 mb-2">Personal Details</h4>
                            <div className="space-y-2">
                                <div>
                                    <p className="text-xs text-gray-500">Full Name</p>
                                    <p className="text-sm text-white">{patient?.name || 'N/A'}</p>
                                </div>
                                <div>
                                    <p className="text-xs text-gray-500">Age</p>
                                    <p className="text-sm text-white">{patient?.age || 'N/A'} years</p>
                                </div>
                                <div>
                                    <p className="text-xs text-gray-500">Gender</p>
                                    <p className="text-sm text-white">{patient?.gender || 'N/A'}</p>
                                </div>
                                <div>
                                    <p className="text-xs text-gray-500">Blood Type</p>
                                    <p className="text-sm text-white">{patient?.bloodType || 'N/A'}</p>
                                </div>
                            </div>
                        </div>

                        <div className="bg-[#0f0f0f] border border-white/5 rounded-lg p-3">
                            <h4 className="text-xs font-semibold text-teal-400 mb-2">Medical Information</h4>
                            <div className="space-y-2">
                                <div>
                                    <p className="text-xs text-gray-500">Diagnosis</p>
                                    <p className="text-sm text-white">{patient?.diagnosis || 'N/A'}</p>
                                </div>
                                <div>
                                    <p className="text-xs text-gray-500">Status</p>
                                    <span className={`inline-block text-xs px-2 py-0.5 rounded ${patient?.status === 'Critical' ? 'bg-red-500/20 text-red-400' :
                                            patient?.status === 'Under Treatment' ? 'bg-yellow-500/20 text-yellow-400' :
                                                'bg-teal-500/20 text-teal-400'
                                        }`}>
                                        {patient?.status || 'N/A'}
                                    </span>
                                </div>
                                <div>
                                    <p className="text-xs text-gray-500">Admission Date</p>
                                    <p className="text-sm text-white">{patient?.admissionDate || 'N/A'}</p>
                                </div>
                                <div>
                                    <p className="text-xs text-gray-500">Last Visit</p>
                                    <p className="text-sm text-white">{patient?.lastVisit || 'N/A'}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
