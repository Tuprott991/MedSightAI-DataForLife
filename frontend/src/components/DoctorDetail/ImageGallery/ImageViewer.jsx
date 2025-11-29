import { useState } from 'react';
import { Undo, Redo, PanelLeft, PanelLeftClose } from 'lucide-react';
import { useSidebar } from '../../layout';
import { SimilarCasesButton } from '../SimilarCases/SimilarCasesButton';
import { SimilarCasesModal } from '../SimilarCases/SimilarCasesModal';
import { ZoomControls } from '../../custom/ZoomControls';
import { ImageToolsSidebar } from './ImageToolsSidebar';

export const ImageViewer = ({ image, patientInfo }) => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [zoom, setZoom] = useState(100);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [brightness, setBrightness] = useState(100);
    const [contrast, setContrast] = useState(100);
    const [activeAdjustment, setActiveAdjustment] = useState(null); // 'brightness' or 'contrast'
    const [activeTool, setActiveTool] = useState(null); // 'square', 'circle', 'freehand'
    const { isLeftCollapsed, setIsLeftCollapsed } = useSidebar();

    const handleZoomIn = () => setZoom(prev => Math.min(prev + 25, 500));
    const handleZoomOut = () => setZoom(prev => Math.max(prev - 25, 50));
    const handleReset = () => {
        setZoom(100);
        setPosition({ x: 0, y: 0 });
        setBrightness(100);
        setContrast(100);
        setActiveAdjustment(null);
        setActiveTool(null);
    };

    const handleBrightnessClick = () => {
        setActiveAdjustment(activeAdjustment === 'brightness' ? null : 'brightness');
    };

    const handleContrastClick = () => {
        setActiveAdjustment(activeAdjustment === 'contrast' ? null : 'contrast');
    };

    const handleToolClick = (tool) => {
        setActiveTool(activeTool === tool ? null : tool);
    };

    if (!image) {
        return (
            <div className="bg-[#1a1a1a] border border-white/10 rounded-xl flex items-center justify-center h-[calc(100vh-110px)]">
                <p className="text-gray-500">Không có hình ảnh được chọn</p>
            </div>
        );
    }

    return (
        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden flex flex-col h-[calc(100vh-110px)]">
            {/* Header with Control Buttons */}
            <div className="px-4 py-2.5 border-b border-white/10 bg-[#141414]">
                <div className="flex items-center justify-between">
                    {/* Group 1: Toggle Sidebar + Zoom and History Controls */}
                    <div className="flex items-center gap-1">
                        {/* Toggle Left Sidebar Button */}
                        <button
                            onClick={() => setIsLeftCollapsed(!isLeftCollapsed)}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors"
                            title={isLeftCollapsed ? "Hiện sidebar trái" : "Ẩn sidebar trái"}
                        >
                            {isLeftCollapsed ? (
                                <PanelLeft className="w-4 h-4" />
                            ) : (
                                <PanelLeftClose className="w-4 h-4" />
                            )}
                        </button>
                        <div className="w-px h-4 bg-white/10 mx-1"></div>

                        <ZoomControls
                            zoom={zoom}
                            onZoomIn={handleZoomIn}
                            onZoomOut={handleZoomOut}
                            onReset={handleReset}
                            minZoom={50}
                            maxZoom={500}
                            showReset={false}
                        />

                        <div className="w-px h-4 bg-white/10 mx-1"></div>
                        <button className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            <Undo className="w-4 h-4" />
                        </button>
                        <button className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            <Redo className="w-4 h-4" />
                        </button>
                    </div>

                    {/* Group 2: Similar Cases Button */}
                    <SimilarCasesButton onClick={() => setIsModalOpen(true)} />
                </div>
            </div>

            {/* Content Area with Inner Sidebar */}
            <div className="flex-1 flex overflow-hidden">
                {/* Inner Sidebar - Tools Panel (Left) */}
                <ImageToolsSidebar
                    activeTool={activeTool}
                    onToolChange={handleToolClick}
                    brightness={brightness}
                    contrast={contrast}
                    activeAdjustment={activeAdjustment}
                    onBrightnessClick={handleBrightnessClick}
                    onContrastClick={handleContrastClick}
                    onReset={handleReset}
                />

                {/* Image Container */}
                <div className="flex-1 flex items-center justify-center bg-black/30 p-4 overflow-hidden">
                    <div
                        style={{
                            transform: `translate(${position.x}px, ${position.y}px) scale(${zoom / 100})`,
                            transition: 'transform 300ms',
                            filter: `brightness(${brightness}%) contrast(${contrast}%)`
                        }}
                    >
                        <img
                            src={image.url}
                            alt={image.type}
                            className="max-w-full max-h-full object-contain"
                        />
                    </div>
                </div>
            </div>

            {/* Bottom Adjustment Slider */}
            {activeAdjustment && (
                <div className="px-4 py-3 border-t border-white/10 bg-[#141414]">
                    <div className="flex items-center gap-4">
                        <span className="text-xs text-gray-400 font-medium min-w-[90px]">
                            {activeAdjustment === 'brightness' ? 'Độ Sáng' : 'Độ Tương Phản'}
                        </span>
                        <input
                            type="range"
                            min="0"
                            max="200"
                            value={activeAdjustment === 'brightness' ? brightness : contrast}
                            onChange={(e) => {
                                const value = Number(e.target.value);
                                if (activeAdjustment === 'brightness') {
                                    setBrightness(value);
                                } else {
                                    setContrast(value);
                                }
                            }}
                            className="flex-1 h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer slider-thumb"
                        />
                        <span className="text-xs text-amber-400 font-semibold min-w-[45px] text-right">
                            {activeAdjustment === 'brightness' ? brightness : contrast}%
                        </span>
                    </div>
                </div>
            )}

            {/* Slider Styles */}
            <style jsx>{`
                .slider-thumb::-webkit-slider-thumb {
                    appearance: none;
                    width: 14px;
                    height: 14px;
                    border-radius: 50%;
                    background: #f59e0b;
                    cursor: pointer;
                    border: 2px solid #0f172a;
                }
                .slider-thumb::-moz-range-thumb {
                    width: 14px;
                    height: 14px;
                    border-radius: 50%;
                    background: #f59e0b;
                    cursor: pointer;
                    border: 2px solid #0f172a;
                }
                .slider-thumb::-webkit-slider-thumb:hover {
                    background: #d97706;
                }
                .slider-thumb::-moz-range-thumb:hover {
                    background: #d97706;
                }
            `}</style>

            {/* Similar Cases Modal */}
            <SimilarCasesModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                currentImage={image}
                patientInfo={patientInfo}
            />
        </div>
    );
};