import { ZoomIn, ZoomOut } from 'lucide-react';

export const ZoomControls = ({
    zoom,
    onZoomIn,
    onZoomOut,
    onReset,
    minZoom = 50,
    maxZoom = 500,
    showReset = true,
    className = ""
}) => {
    return (
        <div className={`flex items-center gap-1 ${className}`}>
            {showReset && (
                <>
                    <button
                        onClick={onReset}
                        className="px-2.5 py-1.5 text-xs text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors cursor-pointer"
                    >
                        Đặt Lại
                    </button>
                    <div className="w-px h-4 bg-white/10 mx-1"></div>
                </>
            )}

            <button
                onClick={onZoomOut}
                disabled={zoom <= minZoom}
                className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                title="Zoom Out"
            >
                <ZoomOut className="w-4 h-4" />
            </button>

            <span className="text-xs text-gray-400 px-2 min-w-[50px] text-center">
                {zoom}%
            </span>

            <button
                onClick={onZoomIn}
                disabled={zoom >= maxZoom}
                className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                title="Zoom In"
            >
                <ZoomIn className="w-4 h-4" />
            </button>
        </div>
    );
};
