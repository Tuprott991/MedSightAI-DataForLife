import { Square, Circle, Pencil, Sun, Contrast, RotateCcw, Eraser, Ruler, RotateCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const ImageToolsSidebar = ({
    activeTool,
    onToolChange,
    brightness,
    contrast,
    activeAdjustment,
    onBrightnessClick,
    onContrastClick,
    onRotateLeft,
    onRotateRight,
    onReset
}) => {
    const { t } = useTranslation();

    return (
        <div className="w-14 border-r border-white/10 bg-[#141414] flex flex-col">
            <div className="flex-1 overflow-y-auto p-1.5 space-y-3">
                {/* Annotation Tools Group */}
                <div>
                    <h4 className="text-[11px] font-semibold text-gray-300 mb-2 text-center">{t('doctorDetail.imageTools.annotationGroup')}</h4>
                    <div className="grid grid-cols-2 gap-1">
                        <button
                            onClick={() => onToolChange('square')}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeTool === 'square'
                                ? 'bg-teal-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title={t('doctorDetail.imageTools.square')}
                        >
                            <Square className="w-4 h-4" />
                        </button>
                        <button
                            onClick={() => onToolChange('circle')}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeTool === 'circle'
                                ? 'bg-teal-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title={t('doctorDetail.imageTools.circle')}
                        >
                            <Circle className="w-4 h-4" />
                        </button>
                        <button
                            onClick={() => onToolChange('freehand')}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeTool === 'freehand'
                                ? 'bg-teal-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title={t('doctorDetail.imageTools.freehand')}
                        >
                            <Pencil className="w-4 h-4" />
                        </button>
                        <button
                            onClick={() => onToolChange('eraser')}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeTool === 'eraser'
                                ? 'bg-red-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title={t('doctorDetail.imageTools.eraser')}
                        >
                            <Eraser className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Divider */}
                <div className="border-t border-white/30"></div>

                {/* Measurement Tools Group */}
                <div>
                    <h4 className="text-[11px] font-semibold text-gray-300 mb-2 text-center">{t('doctorDetail.imageTools.measurementGroup')}</h4>
                    <div className="flex justify-center">
                        <button
                            onClick={() => onToolChange('ruler')}
                            className={`aspect-square w-10 flex items-center justify-center rounded transition-colors ${activeTool === 'ruler'
                                ? 'bg-purple-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title={t('doctorDetail.imageTools.ruler')}
                        >
                            <Ruler className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Divider */}
                <div className="border-t border-white/20"></div>

                {/* Light Adjustment Tools Group */}
                <div>
                    <h4 className="text-[11px] font-semibold text-gray-300 mb-2 text-center">{t('doctorDetail.imageTools.lightGroup')}</h4>
                    <div className="grid grid-cols-2 gap-1">
                        <button
                            onClick={onBrightnessClick}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeAdjustment === 'brightness'
                                ? 'bg-amber-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title={t('doctorDetail.imageTools.brightness')}
                        >
                            <Sun className="w-4 h-4" />
                        </button>
                        <button
                            onClick={onContrastClick}
                            className={`aspect-square flex items-center justify-center rounded transition-colors ${activeAdjustment === 'contrast'
                                ? 'bg-amber-500 text-white'
                                : 'text-gray-300 hover:text-white hover:bg-white/5'
                                }`}
                            title={t('doctorDetail.imageTools.contrast')}
                        >
                            <Contrast className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Divider */}
                <div className="border-t border-white/20"></div>

                {/* Other Utilities Group */}
                <div>
                    <h4 className="text-[11px] font-semibold text-gray-300 mb-2 text-center">{t('doctorDetail.imageTools.utilitiesGroup')}</h4>
                    <div className="grid grid-cols-2 gap-1">
                        <button
                            onClick={onRotateLeft}
                            className="aspect-square flex items-center justify-center rounded transition-colors text-gray-300 hover:text-white hover:bg-white/5"
                            title={t('doctorDetail.imageTools.rotateLeft')}
                        >
                            <RotateCcw className="w-4 h-4" />
                        </button>
                        <button
                            onClick={onRotateRight}
                            className="aspect-square flex items-center justify-center rounded transition-colors text-gray-300 hover:text-white hover:bg-white/5"
                            title={t('doctorDetail.imageTools.rotateRight')}
                        >
                            <RotateCw className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Reset Button */}
                <div className="pt-1.5 border-t border-white/10 text-[11px] font-semibold">
                    <button
                        onClick={onReset}
                        className="w-full py-1 flex items-center justify-center rounded bg-white/5 hover:bg-white/10 text-gray-300 hover:text-white transition-colors cursor-pointer"
                        title={t('doctorDetail.imageTools.resetAll')}
                    >
                        {t('doctorDetail.imageTools.reset')}
                    </button>
                </div>
            </div>
        </div>
    );
};
