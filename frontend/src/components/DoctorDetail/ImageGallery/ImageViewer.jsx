import { useState } from 'react';
import { ZoomIn, ZoomOut, Undo, Redo, PenTool, PanelLeft, PanelLeftClose } from 'lucide-react';
import { useSidebar } from '../../layout';
import { SimilarCasesButton } from '../SimilarCases/SimilarCasesButton';
import { SimilarCasesModal } from '../SimilarCases/SimilarCasesModal';

export const ImageViewer = ({ image, patientInfo }) => {
    const [sliceValue, setSliceValue] = useState(75);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const { isLeftCollapsed, setIsLeftCollapsed } = useSidebar();

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
                        {/* Toggle Sidebar Button */}
                        <button
                            onClick={() => setIsLeftCollapsed(!isLeftCollapsed)}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors"
                            title={isLeftCollapsed ? "Show sidebar" : "Hide sidebar"}
                        >
                            {isLeftCollapsed ? (
                                <PanelLeft className="w-4 h-4" />
                            ) : (
                                <PanelLeftClose className="w-4 h-4" />
                            )}
                        </button>
                        <div className="w-px h-4 bg-white/10 mx-1"></div>

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

                    {/* Group 2: Similar Cases Button */}
                    <SimilarCasesButton onClick={() => setIsModalOpen(true)} />

                    {/* Group 3: Annotate */}
                    <div className="flex items-center gap-1">
                        <button className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors">
                            <PenTool className="w-3.5 h-3.5" />
                            <span>Chú Thích</span>
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

            {/* Bottom Slider */}
            <div className="px-4 py-3 border-t border-white/10 bg-[#141414]">
                <div className="flex items-center gap-4">
                    {/* Slice Label */}
                    <span className="text-xs text-gray-400 font-medium min-w-[70px]">
                        Lớp #{sliceValue}
                    </span>

                    {/* Slider */}
                    <input
                        type="range"
                        min="0"
                        max="150"
                        value={sliceValue}
                        onChange={(e) => setSliceValue(Number(e.target.value))}
                        className="flex-1 h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer slider-thumb"
                    />
                </div>
            </div>

            {/* Slider Styles */}
            <style jsx>{`
                .slider-thumb::-webkit-slider-thumb {
                    appearance: none;
                    width: 14px;
                    height: 14px;
                    border-radius: 50%;
                    background: #14b8a6;
                    cursor: pointer;
                    border: 2px solid #0f172a;
                }
                .slider-thumb::-moz-range-thumb {
                    width: 14px;
                    height: 14px;
                    border-radius: 50%;
                    background: #14b8a6;
                    cursor: pointer;
                    border: 2px solid #0f172a;
                }
                .slider-thumb::-webkit-slider-thumb:hover {
                    background: #0d9488;
                }
                .slider-thumb::-moz-range-thumb:hover {
                    background: #0d9488;
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