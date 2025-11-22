import { useState } from 'react';
import { ZoomIn, ZoomOut, RotateCw, Maximize2 } from 'lucide-react';

export const ImageInteractiveSection = ({ caseData }) => {
    const [zoom, setZoom] = useState(100);
    const [rotation, setRotation] = useState(0);

    const handleZoomIn = () => setZoom(prev => Math.min(prev + 25, 200));
    const handleZoomOut = () => setZoom(prev => Math.max(prev - 25, 50));
    const handleRotate = () => setRotation(prev => (prev + 90) % 360);
    const handleReset = () => {
        setZoom(100);
        setRotation(0);
    };

    if (!caseData?.imageUrl) {
        return (
            <div className="flex-1 min-h-0 bg-[#1a1a1a] border border-white/10 rounded-xl flex items-center justify-center">
                <p className="text-gray-500">No image available</p>
            </div>
        );
    }

    return (
        <div className="flex-1 min-h-0 bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden flex flex-col">
            {/* Toolbar */}
            <div className="px-4 py-2.5 border-b border-white/10 bg-[#141414] shrink-0">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                        <button
                            onClick={handleZoomOut}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors"
                            title="Zoom Out"
                        >
                            <ZoomOut className="w-4 h-4" />
                        </button>
                        <span className="text-xs text-gray-400 px-2 min-w-[50px] text-center">
                            {zoom}%
                        </span>
                        <button
                            onClick={handleZoomIn}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors"
                            title="Zoom In"
                        >
                            <ZoomIn className="w-4 h-4" />
                        </button>

                        <div className="w-px h-4 bg-white/10 mx-2"></div>

                        <button
                            onClick={handleRotate}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors"
                            title="Rotate"
                        >
                            <RotateCw className="w-4 h-4" />
                        </button>
                    </div>

                    <div className="flex items-center gap-1">
                        <button
                            onClick={handleReset}
                            className="px-2.5 py-1.5 text-xs text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors"
                        >
                            Reset
                        </button>
                    </div>
                </div>
            </div>

            {/* Image Display */}
            <div className="flex-1 bg-black/30 flex items-center justify-center overflow-hidden p-4">
                <img
                    src={caseData.imageUrl}
                    alt={caseData.diagnosis || 'Medical Image'}
                    className="max-w-full max-h-full object-contain transition-transform duration-300"
                    style={{
                        transform: `scale(${zoom / 100}) rotate(${rotation}deg)`
                    }}
                />
            </div>

            {/* Case Info */}
            <div className="px-4 py-3 border-t border-white/10 bg-[#141414] shrink-0">
                <div className="flex items-center justify-between text-xs">
                    <div className="text-gray-400">
                        <span className="font-semibold text-white">{caseData.patientName}</span>
                        {caseData.age && caseData.gender && (
                            <span className="ml-2">• {caseData.age}y, {caseData.gender}</span>
                        )}
                    </div>
                    {caseData.diagnosis && (
                        <div className="text-teal-400">{caseData.diagnosis}</div>
                    )}
                </div>
            </div>
        </div>
    );
};
