import { useState, useRef } from 'react';
import { ZoomIn, ZoomOut, Pencil, Trash2, Hand, Undo, Redo } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const ImageInteractiveSection = ({ caseData, onAnnotationsChange }) => {
    const { t } = useTranslation();
    const [zoom, setZoom] = useState(100);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDrawing, setIsDrawing] = useState(false);
    const [isPanning, setIsPanning] = useState(false);
    const [startPoint, setStartPoint] = useState(null);
    const [currentBox, setCurrentBox] = useState(null);
    const [boxes, setBoxes] = useState([]);
    const [history, setHistory] = useState([]);
    const [historyIndex, setHistoryIndex] = useState(-1);
    const [drawMode, setDrawMode] = useState(false);
    const [panMode, setPanMode] = useState(false);
    const [selectedBoxIndex, setSelectedBoxIndex] = useState(null);
    const [editingLabel, setEditingLabel] = useState('');
    const [comparisonImage, setComparisonImage] = useState(null);
    const [showComparison, setShowComparison] = useState(false);
    const containerRef = useRef(null);
    const imageRef = useRef(null);

    const handleZoomIn = () => setZoom(prev => Math.min(prev + 25, 500));
    const handleZoomOut = () => setZoom(prev => Math.max(prev - 25, 50));
    const handleReset = () => {
        setZoom(100);
        setPosition({ x: 0, y: 0 });
    };

    const addToHistory = (newBoxes) => {
        const newHistory = history.slice(0, historyIndex + 1);
        newHistory.push(newBoxes);
        setHistory(newHistory);
        setHistoryIndex(newHistory.length - 1);
    };

    const handleUndo = () => {
        if (historyIndex > 0) {
            const newIndex = historyIndex - 1;
            setHistoryIndex(newIndex);
            const previousState = history[newIndex];
            setBoxes(previousState);
            onAnnotationsChange?.(previousState);
        }
    };

    const handleRedo = () => {
        if (historyIndex < history.length - 1) {
            const newIndex = historyIndex + 1;
            setHistoryIndex(newIndex);
            const nextState = history[newIndex];
            setBoxes(nextState);
            onAnnotationsChange?.(nextState);
        }
    };

    const toggleDrawMode = () => {
        setDrawMode(!drawMode);
        setPanMode(false);
        setIsDrawing(false);
        setStartPoint(null);
        setCurrentBox(null);
    };

    const togglePanMode = () => {
        setPanMode(!panMode);
        setDrawMode(false);
        setIsPanning(false);
        setStartPoint(null);
    };

    const clearAllBoxes = () => {
        const newBoxes = [];
        setBoxes(newBoxes);
        addToHistory(newBoxes);
        onAnnotationsChange?.(newBoxes);
    };

    const handleMouseDown = (e) => {
        // Ignore if clicking on input or button
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') {
            return;
        }

        if (!imageRef.current) return;

        const rect = imageRef.current.getBoundingClientRect();
        const scale = zoom / 100;

        // Convert screen coordinates to image coordinates (accounting for zoom)
        const x = (e.clientX - rect.left) / scale;
        const y = (e.clientY - rect.top) / scale;

        if (panMode) {
            setIsPanning(true);
            setStartPoint({ x: e.clientX - position.x, y: e.clientY - position.y });
        } else if (drawMode) {
            setIsDrawing(true);
            setStartPoint({ x, y });
            setCurrentBox({ x, y, width: 0, height: 0 });
        }
    };

    const handleMouseMove = (e) => {
        if (!imageRef.current) return;

        if (isPanning && startPoint) {
            setPosition({
                x: e.clientX - startPoint.x,
                y: e.clientY - startPoint.y
            });
        } else if (isDrawing && startPoint) {
            const rect = imageRef.current.getBoundingClientRect();
            const scale = zoom / 100;

            // Convert screen coordinates to image coordinates (accounting for zoom)
            const currentX = (e.clientX - rect.left) / scale;
            const currentY = (e.clientY - rect.top) / scale;

            const width = currentX - startPoint.x;
            const height = currentY - startPoint.y;

            setCurrentBox({
                x: width < 0 ? currentX : startPoint.x,
                y: height < 0 ? currentY : startPoint.y,
                width: Math.abs(width),
                height: Math.abs(height)
            });
        }
    };

    const handleMouseUp = () => {
        if (isPanning) {
            setIsPanning(false);
            setStartPoint(null);
        } else if (isDrawing && currentBox) {
            if (currentBox.width > 5 && currentBox.height > 5) {
                const newBox = { ...currentBox, label: 'Phát hiện' };
                const newBoxes = [...boxes, newBox];
                setBoxes(newBoxes);
                addToHistory(newBoxes);
                onAnnotationsChange?.(newBoxes);
            }
            setIsDrawing(false);
            setStartPoint(null);
            setCurrentBox(null);
        }
    };

    const handleBoxClick = (index, e) => {
        e.stopPropagation();
        setSelectedBoxIndex(index);
        setEditingLabel(boxes[index].label || '');
    };

    const handleLabelChange = (e) => {
        setEditingLabel(e.target.value);
    };

    const handleLabelSubmit = (index) => {
        if (editingLabel.trim()) {
            const newBoxes = [...boxes];
            newBoxes[index] = { ...newBoxes[index], label: editingLabel.trim() };
            setBoxes(newBoxes);
            addToHistory(newBoxes);
            onAnnotationsChange?.(newBoxes);
        }
        setSelectedBoxIndex(null);
        setEditingLabel('');
    };

    const handleDeleteBox = (index, e) => {
        e.stopPropagation();
        const newBoxes = boxes.filter((_, i) => i !== index);
        setBoxes(newBoxes);
        addToHistory(newBoxes);
        onAnnotationsChange?.(newBoxes);
        setSelectedBoxIndex(null);
    };

    // Handle drag and drop từ chatbot
    const handleDragOver = (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const imageUrl = e.dataTransfer.getData('imageUrl');
        const imageLabel = e.dataTransfer.getData('imageLabel');

        if (imageUrl) {
            setComparisonImage({ url: imageUrl, label: imageLabel });
            setShowComparison(true);
        }
    };

    const toggleComparison = () => {
        setShowComparison(!showComparison);
    };

    if (!caseData?.imageUrl) {
        return (
            <div className="flex-1 min-h-0 bg-[#1a1a1a] border border-white/10 rounded-xl flex items-center justify-center">
                <p className="text-gray-500">{t('doctor.noResults')}</p>
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
                            onClick={handleReset}
                            className="px-2.5 py-1.5 text-xs text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors cursor-pointer"
                        >
                            {t('studentDetail.interactive.reset')}
                        </button>

                        <div className="w-px h-4 bg-white/10 mx-2"></div>

                        <button
                            onClick={handleZoomOut}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors cursor-pointer"
                            title="Zoom Out"
                        >
                            <ZoomOut className="w-4 h-4" />
                        </button>
                        <span className="text-xs text-gray-400 px-2 min-w-[50px] text-center">
                            {zoom}%
                        </span>
                        <button
                            onClick={handleZoomIn}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors cursor-pointer"
                            title="Zoom In"
                        >
                            <ZoomIn className="w-4 h-4" />
                        </button>

                        <div className="w-px h-4 bg-white/10 mx-2"></div>

                        <button
                            onClick={handleUndo}
                            disabled={historyIndex <= 0}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                            title="Undo"
                        >
                            <Undo className="w-4 h-4" />
                        </button>
                        <button
                            onClick={handleRedo}
                            disabled={historyIndex >= history.length - 1}
                            className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                            title="Redo"
                        >
                            <Redo className="w-4 h-4" />
                        </button>
                    </div>

                    <div className="flex items-center gap-1">
                        <button
                            onClick={togglePanMode}
                            className={`cursor-pointer flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded transition-colors ${panMode
                                ? 'bg-teal-500 text-white'
                                : 'text-gray-400 hover:text-white hover:bg-white/10'
                                }`}
                            title="Pan/Move Image"
                        >
                            <Hand className="w-3.5 h-3.5" />
                            <span>Di Chuyển</span>
                        </button>

                        <button
                            onClick={toggleDrawMode}
                            className={`cursor-pointer flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded transition-colors ${drawMode
                                ? 'bg-teal-500 text-white'
                                : 'text-gray-400 hover:text-white hover:bg-white/10'
                                }`}
                            title="Draw Bounding Box"
                        >
                            <Pencil className="w-3.5 h-3.5" />
                            <span>Vẽ</span>
                        </button>

                        {boxes.length > 0 && (
                            <button
                                onClick={clearAllBoxes}
                                className={`cursor-pointer flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded transition-colors border border-red-500 text-red-500 hover:bg-red-500/10 hover:text-red-400`}
                                title="Delete All Annotations"
                            >
                                <Trash2 className="w-3.5 h-3.5" />
                                <span>Xóa tất cả</span>
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* Image Display */}
            <div
                ref={containerRef}
                className="flex-1 bg-black/30 flex items-center justify-center overflow-hidden p-4"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
            >
                {!showComparison ? (
                    // Single image view
                    <div
                        className="relative"
                        style={{
                            cursor: drawMode ? 'crosshair' : panMode ? 'grab' : 'default',
                            transform: `translate(${position.x}px, ${position.y}px) scale(${zoom / 100})`,
                            transition: isPanning ? 'none' : 'transform 300ms'
                        }}
                    >
                        <img
                            ref={imageRef}
                            src={caseData.imageUrl}
                            alt={caseData.diagnosis || 'Medical Image'}
                            className="max-w-full max-h-full object-contain select-none"
                            onMouseDown={handleMouseDown}
                            onMouseMove={handleMouseMove}
                            onMouseUp={handleMouseUp}
                            onMouseLeave={handleMouseUp}
                            draggable={false}
                        />

                        {/* Render saved boxes */}
                        {boxes.map((box, index) => (
                            <div
                                key={index}
                                onClick={(e) => handleBoxClick(index, e)}
                                className={`absolute border-2 pointer-events-auto cursor-pointer transition-all ${selectedBoxIndex === index
                                    ? 'border-yellow-400 shadow-lg'
                                    : 'border-teal-500'
                                    }`}
                                style={{
                                    left: box.x,
                                    top: box.y,
                                    width: box.width,
                                    height: box.height,
                                    backgroundColor: selectedBoxIndex === index
                                        ? 'rgba(250, 204, 21, 0.15)'
                                        : 'rgba(20, 184, 166, 0.1)'
                                }}
                            >
                                <div
                                    className={`absolute left-0 flex items-center gap-1 px-1.5 py-0.5 rounded whitespace-nowrap ${selectedBoxIndex === index ? 'bg-yellow-400' : 'bg-teal-500'
                                        } text-white`}
                                    style={{
                                        top: -20 / (zoom / 100),
                                        fontSize: `${12 / (zoom / 100)}px`
                                    }}
                                >
                                    {selectedBoxIndex === index ? (
                                        <>
                                            <input
                                                type="text"
                                                value={editingLabel}
                                                onChange={handleLabelChange}
                                                onBlur={() => handleLabelSubmit(index)}
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter') handleLabelSubmit(index);
                                                    if (e.key === 'Escape') setSelectedBoxIndex(null);
                                                    e.stopPropagation();
                                                }}
                                                onClick={(e) => e.stopPropagation()}
                                                className="bg-white/20 px-1 rounded outline-none"
                                                style={{
                                                    width: '80px',
                                                    fontSize: `${12 / (zoom / 100)}px`
                                                }}
                                                autoFocus
                                            />
                                            <button
                                                onClick={(e) => handleDeleteBox(index, e)}
                                                className="hover:bg-white/20 rounded px-0.5"
                                                style={{ fontSize: `${12 / (zoom / 100)}px` }}
                                            >
                                                ×
                                            </button>
                                        </>
                                    ) : (
                                        <span>#{index + 1}: {box.label || 'Finding'}</span>
                                    )}
                                </div>
                            </div>
                        ))}

                        {/* Render current drawing box */}
                        {currentBox && (
                            <div
                                className="absolute border-2 border-yellow-400 pointer-events-none"
                                style={{
                                    left: currentBox.x,
                                    top: currentBox.y,
                                    width: currentBox.width,
                                    height: currentBox.height,
                                    backgroundColor: 'rgba(250, 204, 21, 0.1)'
                                }}
                            />
                        )}
                    </div>
                ) : (
                    // Comparison view - 2 images side by side
                    <div className="w-full h-full grid grid-cols-2 gap-4">
                        {/* Original image */}
                        <div className="relative flex flex-col border border-white/20 rounded-lg overflow-hidden">
                            <div className="bg-white/10 px-3 py-2 text-sm font-semibold text-white">
                                Ảnh gốc - Kết quả của bạn
                            </div>
                            <div className="flex-1 flex items-center justify-center bg-black/30 p-2">
                                <div className="relative">
                                    <img
                                        src={caseData.imageUrl}
                                        alt="Original"
                                        className="max-w-full max-h-full object-contain"
                                    />
                                    {/* Render boxes on original */}
                                    {boxes.map((box, index) => (
                                        <div
                                            key={index}
                                            className="absolute border-2 border-teal-500"
                                            style={{
                                                left: box.x,
                                                top: box.y,
                                                width: box.width,
                                                height: box.height,
                                                backgroundColor: 'rgba(20, 184, 166, 0.1)'
                                            }}
                                        >
                                            <div className="absolute left-0 top-[-18px] px-1.5 py-0.5 bg-teal-500 text-white text-[10px] rounded whitespace-nowrap">
                                                #{index + 1}: {box.label}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Kết quả thực tế image */}
                        <div className="relative flex flex-col border border-white/20 rounded-lg overflow-hidden">
                            <div className="bg-white/10 px-3 py-2 text-sm font-semibold text-white">
                                {comparisonImage?.label || 'Kết quả thực tế'}
                            </div>
                            <div className="flex-1 flex items-center justify-center bg-black/30 p-2">
                                <img
                                    src={comparisonImage?.url}
                                    alt="Kết quả thực tế"
                                    className="max-w-full max-h-full object-contain"
                                />
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Case Info */}
            <div className="px-4 py-3 border-t border-white/10 bg-[#141414] shrink-0">
                <div className="flex items-center justify-end text-xs">
                    <div className="flex items-center gap-3">
                        {boxes.length > 0 && (
                            <span className="text-teal-400">{boxes.length} {t('studentDetail.interactive.regions')}</span>
                        )}
                        {comparisonImage && (
                            <button
                                onClick={toggleComparison}
                                className="px-2 py-1 bg-teal-500/20 hover:bg-teal-500/30 text-teal-400 rounded text-xs transition-colors"
                            >
                                {showComparison ? t('common.hide') : t('studentDetail.interactive.toggleComparison')}
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};