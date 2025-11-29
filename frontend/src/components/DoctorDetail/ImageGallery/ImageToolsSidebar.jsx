import { Square, Circle, Pencil, Sun, Contrast, RotateCcw, Eraser } from 'lucide-react';

export const ImageToolsSidebar = ({
    activeTool,
    onToolChange,
    brightness,
    contrast,
    activeAdjustment,
    onBrightnessClick,
    onContrastClick,
    onReset
}) => {
    return (
        <div className="w-14 border-r border-white/10 bg-[#141414] flex flex-col">
            <div className="flex-1 overflow-y-auto p-1.5 space-y-2">
                {/* Annotation Tools Group */}
                <div>
                    <h4 className="text-[11px] font-semibold text-gray-400 mb-1.5 text-center">Khoanh Vùng</h4>
                    <div className="grid grid-cols-2 gap-1">
                        <button
                            onClick={() => onToolChange('square')}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeTool === 'square'
                                ? 'bg-teal-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title="Hình Vuông"
                        >
                            <Square className="w-4 h-4" />
                        </button>
                        <button
                            onClick={() => onToolChange('circle')}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeTool === 'circle'
                                ? 'bg-teal-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title="Hình Tròn"
                        >
                            <Circle className="w-4 h-4" />
                        </button>
                        <button
                            onClick={() => onToolChange('freehand')}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeTool === 'freehand'
                                ? 'bg-teal-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title="Tự Do"
                        >
                            <Pencil className="w-4 h-4" />
                        </button>
                        <button
                            onClick={() => onToolChange('eraser')}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeTool === 'eraser'
                                ? 'bg-red-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title="Xóa"
                        >
                            <Eraser className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Image Adjustment Tools Group */}
                <div>
                    <h4 className="text-[11px] font-semibold text-gray-400 mb-1.5 text-center">Điều Chỉnh</h4>
                    <div className="grid grid-cols-2 gap-1">
                        <button
                            onClick={onBrightnessClick}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeAdjustment === 'brightness'
                                ? 'bg-amber-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title="Độ Sáng"
                        >
                            <Sun className="w-4 h-4" />
                        </button>
                        <button
                            onClick={onContrastClick}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeAdjustment === 'contrast'
                                ? 'bg-amber-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title="Độ Tương Phản"
                        >
                            <Contrast className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Reset Button */}
                <div className="pt-1.5 border-t border-white/10">
                    <button
                        onClick={onReset}
                        className="w-full aspect-square flex items-center justify-center rounded bg-white/5 hover:bg-white/10 text-gray-300 hover:text-white transition-colors"
                        title="Đặt Lại"
                    >
                        <RotateCcw className="w-4 h-4" />
                    </button>
                </div>
            </div>
        </div>
    );
};
