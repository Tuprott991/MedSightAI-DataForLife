import { ZoomIn, ZoomOut, Undo, Redo, PenTool } from 'lucide-react';

export const ImageViewer = ({ image }) => {
    if (!image) {
        return (
            <div className="bg-[#1a1a1a] border border-white/10 rounded-xl flex items-center justify-center h-[calc(100vh-110px)]">
                <p className="text-gray-500">No image selected</p>
            </div>
        );
    }

    return (
        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden flex flex-col h-[calc(100vh-110px)]">
            {/* Header with Control Buttons */}
            <div className="px-4 py-2.5 border-b border-white/10 bg-[#141414]">
                <div className="flex items-center justify-between">
                    {/* Group 1: Zoom and History Controls */}
                    <div className="flex items-center gap-1">
                        <button className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            <ZoomIn className="w-4 h-4" />
                        </button>
                        <button className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            <ZoomOut className="w-4 h-4" />
                        </button>
                        <div className="w-px h-4 bg-white/10 mx-1"></div>
                        <button className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            <Undo className="w-4 h-4" />
                        </button>
                        <button className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            <Redo className="w-4 h-4" />
                        </button>
                    </div>

                    {/* Group 2: Viewing Modes */}
                    <div className="flex items-center gap-1">
                        <button className="px-2.5 py-1.5 text-xs text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            MIP
                        </button>
                        <button className="px-2.5 py-1.5 text-xs text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            MPR
                        </button>
                        <button className="px-2.5 py-1.5 text-xs text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            qCT
                        </button>
                    </div>

                    {/* Group 3: Annotate */}
                    <div className="flex items-center gap-1">
                        <button className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            <PenTool className="w-3.5 h-3.5" />
                            <span>Annotate</span>
                        </button>
                    </div>
                </div>
            </div>

            {/* Image Container */}
            <div className="flex-1 flex items-center justify-center bg-black/30 p-4 overflow-hidden">
                <img
                    src={image.url}
                    alt={image.type}
                    className="max-w-full max-h-full object-contain"
                />
            </div>
        </div>
    );
};