import { useState } from 'react';
import { ChevronDown, ChevronRight, Calendar, Folder } from 'lucide-react';

export const ImageListGrouped = ({ imageGroups, selectedImage, onImageSelect }) => {
    const [expandedGroups, setExpandedGroups] = useState([imageGroups[0]?.id]);

    const toggleGroup = (groupId) => {
        setExpandedGroups(prev =>
            prev.includes(groupId)
                ? prev.filter(id => id !== groupId)
                : [...prev, groupId]
        );
    };

    const isExpanded = (groupId) => expandedGroups.includes(groupId);

    return (
        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden flex flex-col h-[calc(100vh-110px)]">
            <div className="px-4 py-3 border-b border-white/10 bg-[#141414]">
                <h3 className="text-base font-semibold text-white flex items-center gap-2">
                    <Folder className="w-4 h-4 text-teal-500" />
                    Medical Images
                </h3>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar p-3">
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
            </div>
        </div>
    );
};
