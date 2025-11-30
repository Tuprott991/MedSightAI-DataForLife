import { useState, useRef, useEffect } from 'react';
import { Undo, Redo, PanelLeft, PanelLeftClose, Hand, PanelRight, PanelRightClose } from 'lucide-react';
import { useSidebar } from '../../layout';
import { SimilarCasesButton } from '../SimilarCases/SimilarCasesButton';
import { SimilarCasesModal } from '../SimilarCases/SimilarCasesModal';
import { ZoomControls } from '../../custom/ZoomControls';
import { ConfirmModal } from '../../custom/ConfirmModal';
import { ImageToolsSidebar } from './ImageToolsSidebar';

export const ImageViewer = ({ image, patientInfo, onRestoreOriginal, onSimilarCaseModeChange, onSimilarCaseDataChange, onImageChange }) => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [zoom, setZoom] = useState(100);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [rotation, setRotation] = useState(0);
    const [brightness, setBrightness] = useState(100);
    const [contrast, setContrast] = useState(100);
    const [activeAdjustment, setActiveAdjustment] = useState(null); // 'brightness' or 'contrast'
    const [activeTool, setActiveTool] = useState(null); // 'square', 'circle', 'freehand', 'eraser', 'ruler'
    const [isPanMode, setIsPanMode] = useState(false);
    const [isPrototypeCollapsed, setIsPrototypeCollapsed] = useState(false);
    const [showPrototype, setShowPrototype] = useState(false); // Toggle between Original and Prototype for right image
    const { isLeftCollapsed, setIsLeftCollapsed } = useSidebar();
    const [comparisonImages, setComparisonImages] = useState(null);

    // Drawing states - separate for single and multiple image modes
    const [singleImageAnnotations, setSingleImageAnnotations] = useState([]);
    const [multipleImageAnnotations, setMultipleImageAnnotations] = useState([]);
    
    // Ruler/measurement states
    const [measurements, setMeasurements] = useState([]);
    const [currentMeasurement, setCurrentMeasurement] = useState(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [startPoint, setStartPoint] = useState(null);
    const [currentShape, setCurrentShape] = useState(null);
    const [selectedAnnotation, setSelectedAnnotation] = useState(null);
    const [editingLabel, setEditingLabel] = useState('');
    const [confirmDelete, setConfirmDelete] = useState(false);
    const [annotationToDelete, setAnnotationToDelete] = useState(null);
    const [measurementToDelete, setMeasurementToDelete] = useState(null);
    const [isPanning, setIsPanning] = useState(false);
    const [panStart, setPanStart] = useState({ x: 0, y: 0 });
    const imageContainerRef = useRef(null);
    const imageRef = useRef(null);

    // Check if image is an array (multiple images from finding click) or single image
    const images = comparisonImages || (Array.isArray(image) ? image : (image ? [image] : []));
    const isMultipleImages = images.length > 1;

    // Use the appropriate annotations based on current mode
    const annotations = isMultipleImages ? multipleImageAnnotations : singleImageAnnotations;
    const setAnnotations = isMultipleImages ? setMultipleImageAnnotations : setSingleImageAnnotations;

    // Reset selected annotation when switching modes
    useEffect(() => {
        setSelectedAnnotation(null);
        setEditingLabel('');
    }, [isMultipleImages]);

    // Zoom limits: 50% to 500% for all modes
    const handleZoomIn = () => setZoom(prev => Math.min(prev + 25, 500));
    const handleZoomOut = () => setZoom(prev => Math.max(prev - 25, 50));
    const handleReset = () => {
        setZoom(100);
        setPosition({ x: 0, y: 0 });
        setRotation(0);
        setBrightness(100);
        setContrast(100);
        setActiveAdjustment(null);
        setActiveTool(null);
        setIsPanMode(false);
        
        // Exit similar case mode if active
        if (comparisonImages) {
            setComparisonImages(null);
            if (onSimilarCaseModeChange) {
                onSimilarCaseModeChange(false);
            }
            if (onSimilarCaseDataChange) {
                onSimilarCaseDataChange(null);
            }
        }
        setSingleImageAnnotations([]);
        setMultipleImageAnnotations([]);
        setSelectedAnnotation(null);
        setEditingLabel('');
        setComparisonImages(null);
        setMeasurements([]);
        setCurrentMeasurement(null);
    };

    const handleCompareImages = (images, caseData) => {
        setComparisonImages(images);
        if (onSimilarCaseModeChange) {
            onSimilarCaseModeChange(true);
        }
        if (onSimilarCaseDataChange) {
            onSimilarCaseDataChange(caseData);
        }
    };



    const handleBrightnessClick = () => {
        setActiveAdjustment(activeAdjustment === 'brightness' ? null : 'brightness');
        setActiveTool(null);
        setIsPanMode(false);
    };

    const handleContrastClick = () => {
        setActiveAdjustment(activeAdjustment === 'contrast' ? null : 'contrast');
        setActiveTool(null);
        setIsPanMode(false);
    };

    const handleToolClick = (tool) => {
        if (tool === 'eraser') {
            // Eraser becomes an active selection tool
            setActiveTool(activeTool === 'eraser' ? null : 'eraser');
        } else {
            setActiveTool(activeTool === tool ? null : tool);
        }
        setActiveAdjustment(null);
        setIsPanMode(false);
    };

    const handleRotateLeft = () => {
        setRotation(prev => prev - 90);
    };

    const handleRotateRight = () => {
        setRotation(prev => prev + 90);
    };

    // Wheel zoom handler - zoom centered on mouse position
    const handleWheel = (e) => {
        e.preventDefault();

        if (!imageContainerRef.current) return;

        const delta = e.deltaY > 0 ? -25 : 25; // Scroll down = zoom out, scroll up = zoom in
        const newZoom = Math.max(50, Math.min(500, zoom + delta));

        if (newZoom === zoom) return; // No change in zoom

        // Get container bounds
        const containerRect = imageContainerRef.current.getBoundingClientRect();

        // Mouse position relative to container center
        const mouseX = e.clientX - containerRect.left - containerRect.width / 2;
        const mouseY = e.clientY - containerRect.top - containerRect.height / 2;

        // Calculate scale factor
        const scaleFactor = newZoom / zoom - 1;

        // Adjust position to zoom toward mouse position
        const newX = position.x - mouseX * scaleFactor;
        const newY = position.y - mouseY * scaleFactor;

        setZoom(newZoom);
        setPosition({ x: newX, y: newY });
    };

    // Drawing functions
    const handleMouseDown = (e) => {
        // Handle pan mode
        if (isPanMode) {
            setIsPanning(true);
            setPanStart({ x: e.clientX - position.x, y: e.clientY - position.y });
            return;
        }

        // Only draw if a tool is active and it's not eraser
        if (!activeTool || activeTool === 'eraser') return;

        // Only allow drawing on left image when multiple images
        if (isMultipleImages && e.currentTarget !== imageContainerRef.current) return;

        if (!imageRef.current) return;

        // Get the bounding rect of the actual image element
        const imgRect = imageRef.current.getBoundingClientRect();

        // Get the original image dimensions
        const imgWidth = imageRef.current.naturalWidth || imageRef.current.width;
        const imgHeight = imageRef.current.naturalHeight || imageRef.current.height;

        // Get rendered dimensions
        const renderedWidth = imgRect.width;
        const renderedHeight = imgRect.height;

        // Calculate scale factors
        const scaleX = imgWidth / renderedWidth;
        const scaleY = imgHeight / renderedHeight;

        // Calculate position relative to the image itself and convert to original image coordinates
        const x = (e.clientX - imgRect.left) * scaleX;
        const y = (e.clientY - imgRect.top) * scaleY;

        setIsDrawing(true);
        setStartPoint({ x, y });

        if (activeTool === 'ruler') {
            setCurrentMeasurement({
                start: { x, y },
                end: { x, y }
            });
        } else if (activeTool === 'freehand') {
            setCurrentShape({
                type: 'freehand',
                points: [{ x, y }]
            });
        } else {
            setCurrentShape({
                type: activeTool,
                x,
                y,
                width: 0,
                height: 0
            });
        }
    };

    const handleMouseMove = (e) => {
        // Handle panning
        if (isPanning) {
            setPosition({
                x: e.clientX - panStart.x,
                y: e.clientY - panStart.y
            });
            return;
        }

        if (!isDrawing || !startPoint || !imageRef.current) return;

        // Get the bounding rect of the actual image element
        const imgRect = imageRef.current.getBoundingClientRect();

        // Get the original image dimensions
        const imgWidth = imageRef.current.naturalWidth || imageRef.current.width;
        const imgHeight = imageRef.current.naturalHeight || imageRef.current.height;

        // Get rendered dimensions
        const renderedWidth = imgRect.width;
        const renderedHeight = imgRect.height;

        // Calculate scale factors
        const scaleX = imgWidth / renderedWidth;
        const scaleY = imgHeight / renderedHeight;

        // Calculate position relative to the image itself and convert to original image coordinates
        const currentX = (e.clientX - imgRect.left) * scaleX;
        const currentY = (e.clientY - imgRect.top) * scaleY;

        if (activeTool === 'ruler') {
            setCurrentMeasurement(prev => ({
                ...prev,
                end: { x: currentX, y: currentY }
            }));
        } else if (activeTool === 'freehand') {
            setCurrentShape(prev => ({
                ...prev,
                points: [...prev.points, { x: currentX, y: currentY }]
            }));
        } else if (activeTool === 'square' || activeTool === 'circle') {
            const width = currentX - startPoint.x;
            const height = currentY - startPoint.y;

            setCurrentShape({
                type: activeTool,
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
            return;
        }

        if (isDrawing && activeTool === 'ruler' && currentMeasurement) {
            // Calculate distance
            const dx = currentMeasurement.end.x - currentMeasurement.start.x;
            const dy = currentMeasurement.end.y - currentMeasurement.start.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Only save if distance is meaningful (> 5 pixels)
            if (distance > 5) {
                setMeasurements(prev => [...prev, currentMeasurement]);
            }
            setCurrentMeasurement(null);
        } else if (isDrawing && currentShape) {
            const newShape = { ...currentShape, label: 'Phát hiện' };
            if (activeTool === 'freehand' && currentShape.points?.length > 2) {
                setAnnotations(prev => [...prev, newShape]);
            } else if ((activeTool === 'square' || activeTool === 'circle') &&
                currentShape.width > 5 && currentShape.height > 5) {
                setAnnotations(prev => [...prev, newShape]);
            }
        }
        setIsDrawing(false);
        setStartPoint(null);
        setCurrentShape(null);
    };

    const handleAnnotationClick = (index, e) => {
        e.stopPropagation();

        // If eraser is active, prompt for deletion confirmation
        if (activeTool === 'eraser') {
            setAnnotationToDelete(index);
            setConfirmDelete(true);
            return;
        }

        // Otherwise, select for editing
        setSelectedAnnotation(index);
        setEditingLabel(annotations[index].label || '');
    };

    const handleLabelChange = (e) => {
        setEditingLabel(e.target.value);
    };

    const handleLabelSubmit = (index) => {
        if (editingLabel.trim()) {
            const newAnnotations = [...annotations];
            newAnnotations[index] = { ...newAnnotations[index], label: editingLabel.trim() };
            setAnnotations(newAnnotations);
        }
        setSelectedAnnotation(null);
        setEditingLabel('');
    };

    const handleDeleteAnnotation = (index, e) => {
        e.stopPropagation();
        const newAnnotations = annotations.filter((_, i) => i !== index);
        setAnnotations(newAnnotations);
        setSelectedAnnotation(null);
    };

    const handleMeasurementClick = (index, e) => {
        e.stopPropagation();
        
        // If eraser is active, delete measurement directly without confirmation
        if (activeTool === 'eraser') {
            const newMeasurements = measurements.filter((_, i) => i !== index);
            setMeasurements(newMeasurements);
        }
    };

    const handleConfirmDelete = () => {
        if (annotationToDelete !== null) {
            const newAnnotations = annotations.filter((_, i) => i !== annotationToDelete);
            setAnnotations(newAnnotations);
            setSelectedAnnotation(null);
        }
        setConfirmDelete(false);
        setAnnotationToDelete(null);
        setMeasurementToDelete(null);
    };

    const handleCancelDelete = () => {
        setConfirmDelete(false);
        setAnnotationToDelete(null);
        setMeasurementToDelete(null);
    };

    // Render measurement line
    const renderMeasurement = (measurement, index, isTemp = false) => {
        const dx = measurement.end.x - measurement.start.x;
        const dy = measurement.end.y - measurement.start.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Calculate midpoint for label
        const midX = (measurement.start.x + measurement.end.x) / 2;
        const midY = (measurement.start.y + measurement.end.y) / 2;
        
        // Calculate angle for perpendicular label offset
        const angle = Math.atan2(dy, dx);
        const offsetX = -Math.sin(angle) * 20;
        const offsetY = Math.cos(angle) * 20;
        
        const labelX = midX + offsetX;
        const labelY = midY + offsetY;
        
        // Format distance (assuming 1 pixel = 0.01cm for medical imaging)
        const distanceCM = (distance * 0.01).toFixed(2);
        
        return (
            <g 
                key={`measurement-${index}`}
                onClick={!isTemp ? (e) => handleMeasurementClick(index, e) : undefined}
                style={{ cursor: activeTool === 'eraser' && !isTemp ? 'pointer' : 'default' }}
            >
                {/* Invisible wider line for easier clicking */}
                <line
                    x1={measurement.start.x}
                    y1={measurement.start.y}
                    x2={measurement.end.x}
                    y2={measurement.end.y}
                    stroke="transparent"
                    strokeWidth={20}
                    strokeLinecap="round"
                    pointerEvents="stroke"
                />
                {/* Visible line */}
                <line
                    x1={measurement.start.x}
                    y1={measurement.start.y}
                    x2={measurement.end.x}
                    y2={measurement.end.y}
                    stroke={isTemp ? '#5eead4' : '#14b8a6'}
                    strokeWidth={2}
                    strokeLinecap="round"
                    pointerEvents="none"
                />
                {/* Start point */}
                <circle
                    cx={measurement.start.x}
                    cy={measurement.start.y}
                    r={4}
                    fill={isTemp ? '#5eead4' : '#14b8a6'}
                />
                {/* End point */}
                <circle
                    cx={measurement.end.x}
                    cy={measurement.end.y}
                    r={4}
                    fill={isTemp ? '#5eead4' : '#14b8a6'}
                />
                {/* Distance label */}
                {!isTemp && distance > 5 && (
                    <g>
                        <rect
                            x={labelX - 25}
                            y={labelY - 12}
                            width={50}
                            height={20}
                            fill="#14b8a6"
                            rx={3}
                        />
                        <text
                            x={labelX}
                            y={labelY + 3}
                            fill="white"
                            fontSize="11"
                            fontWeight="600"
                            textAnchor="middle"
                        >
                            {distanceCM} cm
                        </text>
                    </g>
                )}
            </g>
        );
    };

    // Render annotation shape with label
    const renderShape = (shape, index, isTemp = false) => {
        const isSelected = index === selectedAnnotation;
        const strokeColor = isTemp ? '#fbbf24' : (isSelected ? '#fbbf24' : '#14b8a6');
        const strokeWidth = 2;

        // Calculate label position based on shape type
        let labelX = 0;
        let labelY = 0;

        if (shape.type === 'square') {
            labelX = shape.x;
            labelY = shape.y - 5;
        } else if (shape.type === 'circle') {
            labelX = shape.x;
            labelY = shape.y - 5;
        } else if (shape.type === 'freehand' && shape.points?.length > 0) {
            labelX = shape.points[0].x;
            labelY = shape.points[0].y - 5;
        }

        return (
            <g>
                {/* Shape */}
                {shape.type === 'square' && (
                    <rect
                        x={shape.x}
                        y={shape.y}
                        width={shape.width}
                        height={shape.height}
                        fill={isSelected ? 'rgba(251, 191, 36, 0.1)' : 'none'}
                        stroke={strokeColor}
                        strokeWidth={strokeWidth}
                        className={!isTemp ? 'cursor-pointer' : ''}
                        style={{ pointerEvents: isTemp ? 'none' : 'auto' }}
                        onClick={(e) => !isTemp && handleAnnotationClick(index, e)}
                    />
                )}
                {shape.type === 'circle' && (
                    <ellipse
                        cx={shape.x + shape.width / 2}
                        cy={shape.y + shape.height / 2}
                        rx={Math.abs(shape.width / 2)}
                        ry={Math.abs(shape.height / 2)}
                        fill={isSelected ? 'rgba(251, 191, 36, 0.1)' : 'none'}
                        stroke={strokeColor}
                        strokeWidth={strokeWidth}
                        className={!isTemp ? 'cursor-pointer' : ''}
                        style={{ pointerEvents: isTemp ? 'none' : 'auto' }}
                        onClick={(e) => !isTemp && handleAnnotationClick(index, e)}
                    />
                )}
                {shape.type === 'freehand' && shape.points?.length > 1 && (
                    <path
                        d={shape.points.map((point, i) =>
                            `${i === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
                        ).join(' ')}
                        fill="none"
                        stroke={strokeColor}
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className={!isTemp ? 'cursor-pointer' : ''}
                        style={{ pointerEvents: isTemp ? 'none' : 'auto' }}
                        onClick={(e) => !isTemp && handleAnnotationClick(index, e)}
                    />
                )}

                {/* Label */}
                {!isTemp && shape.label && (
                    <g>
                        <rect
                            x={labelX}
                            y={labelY - 18}
                            width={isSelected ? 120 : (shape.label.length * 7 + 10)}
                            height={20}
                            fill={strokeColor}
                            rx={3}
                            className={!isSelected ? 'cursor-pointer' : ''}
                            style={{ pointerEvents: 'auto' }}
                            onClick={(e) => !isSelected && handleAnnotationClick(index, e)}
                        />
                        {isSelected ? (
                            <foreignObject
                                x={labelX + 2}
                                y={labelY - 16}
                                width={116}
                                height={18}
                            >
                                <input
                                    type="text"
                                    value={editingLabel}
                                    onChange={handleLabelChange}
                                    onBlur={() => handleLabelSubmit(index)}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter') handleLabelSubmit(index);
                                        if (e.key === 'Escape') setSelectedAnnotation(null);
                                        e.stopPropagation();
                                    }}
                                    onClick={(e) => e.stopPropagation()}
                                    className="w-full h-full bg-transparent text-white text-xs px-1 outline-none"
                                    style={{ pointerEvents: 'auto' }}
                                    autoFocus
                                />
                            </foreignObject>
                        ) : (
                            <text
                                x={labelX + 5}
                                y={labelY - 5}
                                fill="white"
                                fontSize="12"
                                fontWeight="500"
                                className="cursor-pointer"
                                style={{ pointerEvents: 'auto' }}
                                onClick={(e) => handleAnnotationClick(index, e)}
                            >
                                {shape.label}
                            </text>
                        )}
                        {isSelected && (
                            <g
                                onClick={(e) => handleDeleteAnnotation(index, e)}
                                className="cursor-pointer"
                                style={{ pointerEvents: 'auto' }}
                            >
                                <circle
                                    cx={labelX + 110}
                                    cy={labelY - 8}
                                    r="8"
                                    fill="rgba(239, 68, 68, 0.9)"
                                />
                                <text
                                    x={labelX + 110}
                                    y={labelY - 4}
                                    fill="white"
                                    fontSize="12"
                                    fontWeight="bold"
                                    textAnchor="middle"
                                >
                                    ×
                                </text>
                            </g>
                        )}
                    </g>
                )}
            </g>
        );
    };

    if (!image || (Array.isArray(image) && image.length === 0)) {
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
                        <button
                            onClick={() => {
                                setIsPanMode(!isPanMode);
                                setActiveTool(null);
                                setActiveAdjustment(null);
                            }}
                            className={`p-1.5 rounded transition-colors ${isPanMode
                                ? 'bg-blue-500 text-white'
                                : 'text-gray-400 hover:text-white hover:bg-white/10'
                                }`}
                            title="Di Chuyển Ảnh"
                        >
                            <Hand className="w-4 h-4" />
                        </button>
                    </div>

                    {/* Group 2: Similar Cases Button (Center) */}
                    <SimilarCasesButton onClick={() => setIsModalOpen(true)} />

                    {/* Group 3: Original Image Button */}
                    <button
                        onClick={() => {
                            if (comparisonImages) {
                                setComparisonImages(null);
                            } else {
                                onRestoreOriginal();
                            }
                        }}
                        disabled={!isMultipleImages}
                        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded transition-all font-medium ${isMultipleImages
                            ? 'bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-500/50 text-gray-300 hover:text-white cursor-pointer'
                            : 'bg-white/5 border border-white/10 text-gray-600 cursor-not-allowed opacity-50'
                            }`}
                        title={isMultipleImages ? "Quay lại hình ảnh gốc" : "Chỉ khả dụng khi xem 2 ảnh"}
                    >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <span>Hình Ảnh Gốc</span>
                    </button>
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
                    onRotateLeft={handleRotateLeft}
                    onRotateRight={handleRotateRight}
                    onReset={handleReset}
                />

                {/* Image Container */}
                <div className="flex-1 flex items-center justify-center bg-black/30 p-4 overflow-hidden relative">
                    {isMultipleImages ? (
                        <div className="flex items-center justify-center w-full h-full gap-0 relative">
                            {images.map((img, index) => (
                                // Hide prototype (index 1) if collapsed
                                (!isPrototypeCollapsed || index === 0) && (
                                    <div
                                        key={img.id || index}
                                        className="flex-1 flex flex-col h-full relative overflow-hidden"
                                    >
                                        {/* Label */}
                                        <div className="flex items-center justify-center py-2 border-b border-white/10 relative">
                                            {index === 0 ? (
                                                // Left image label - xAI
                                                <span className="text-sm font-semibold text-teal-400">
                                                    {comparisonImages ? 'Ảnh của bệnh nhân' : 'xAI'}
                                                </span>
                                            ) : comparisonImages ? (
                                                // Similar case comparison label
                                                <span className="text-sm font-semibold text-amber-400">
                                                    Ca bệnh tương đồng
                                                </span>
                                            ) : (
                                                // Right image label - Original | Prototype toggle
                                                <div className="flex items-center gap-1">
                                                    <button
                                                        onClick={() => setShowPrototype(false)}
                                                        className={`px-3 py-1 text-xs font-semibold rounded transition-colors ${
                                                            !showPrototype
                                                                ? 'bg-amber-500 text-white'
                                                                : 'text-amber-400 hover:bg-amber-500/20'
                                                        }`}
                                                    >
                                                        Original
                                                    </button>
                                                    <span className="text-gray-500">|</span>
                                                    <button
                                                        onClick={() => setShowPrototype(true)}
                                                        className={`px-3 py-1 text-xs font-semibold rounded transition-colors ${
                                                            showPrototype
                                                                ? 'bg-amber-500 text-white'
                                                                : 'text-amber-400 hover:bg-amber-500/20'
                                                        }`}
                                                    >
                                                        Prototype
                                                    </button>
                                                </div>
                                            )}

                                            {/* Collapse/Expand Button */}
                                            {index === 1 && (
                                                <button
                                                    onClick={() => setIsPrototypeCollapsed(!isPrototypeCollapsed)}
                                                    className="absolute right-2 p-1 hover:bg-white/10 rounded text-gray-400 hover:text-white transition-colors"
                                                    title="Ẩn panel phải"
                                                >
                                                    <PanelRightClose className="w-4 h-4" />
                                                </button>
                                            )}
                                            {index === 0 && isPrototypeCollapsed && (
                                                <button
                                                    onClick={() => setIsPrototypeCollapsed(false)}
                                                    className="absolute right-2 p-1 hover:bg-white/10 rounded text-gray-400 hover:text-white transition-colors"
                                                    title="Hiện panel phải"
                                                >
                                                    <PanelRight className="w-4 h-4" />
                                                </button>
                                            )}
                                        </div>

                                        {/* Image Area */}
                                        <div
                                            className="flex-1 flex items-center justify-center relative overflow-hidden"
                                            ref={index === 0 ? imageContainerRef : null}
                                            onMouseDown={index === 0 ? handleMouseDown : undefined}
                                            onMouseMove={index === 0 ? handleMouseMove : undefined}
                                            onMouseUp={index === 0 ? handleMouseUp : undefined}
                                            onMouseLeave={index === 0 ? handleMouseUp : undefined}
                                            onWheel={index === 0 ? handleWheel : undefined}
                                            style={{
                                                cursor: index === 0
                                                    ? (isPanMode ? (isPanning ? 'grabbing' : 'grab') : (activeTool && activeTool !== 'eraser' ? 'crosshair' : 'default'))
                                                    : 'default'
                                            }}
                                        >
                                            <div
                                                className="relative"
                                                style={{
                                                    transform: `translate(${position.x}px, ${position.y}px) rotate(${rotation}deg) scale(${zoom / 100})`,
                                                    transition: isPanning ? 'none' : 'transform 300ms'
                                                }}
                                            >
                                                <img
                                                    ref={index === 0 ? imageRef : null}
                                                    src={
                                                        index === 1 && img.original && img.prototype
                                                            ? (showPrototype ? (img.prototype.url || img.original.url) : img.original.url)
                                                            : img.url
                                                    }
                                                    alt={img.type}
                                                    className="max-w-full max-h-full object-contain select-none"
                                                    style={{ filter: index === 0 ? `brightness(${brightness}%) contrast(${contrast}%)` : 'none' }}
                                                    draggable={false}
                                                />

                                                {/* SVG overlay for annotations (only on left image) */}
                                                {index === 0 && imageRef.current && (
                                                    <svg
                                                        className="absolute top-0 left-0 w-full h-full"
                                                        viewBox={`0 0 ${imageRef.current.naturalWidth || imageRef.current.width} ${imageRef.current.naturalHeight || imageRef.current.height}`}
                                                        preserveAspectRatio="none"
                                                        style={{ overflow: 'visible' }}
                                                    >
                                                        {/* Render saved annotations */}
                                                        {annotations.map((shape, idx) => (
                                                            <g key={idx}>{renderShape(shape, idx, false)}</g>
                                                        ))}
                                                        {/* Render current drawing shape */}
                                                        {currentShape && renderShape(currentShape, -1, true)}
                                                        {/* Render saved measurements */}
                                                        {measurements.map((measurement, idx) => 
                                                            renderMeasurement(measurement, idx, false)
                                                        )}
                                                        {/* Render current measurement */}
                                                        {currentMeasurement && renderMeasurement(currentMeasurement, -1, true)}
                                                    </svg>
                                                )}
                                            </div>

                                            {index === 0 && !isPrototypeCollapsed && (
                                                <div className="absolute right-0 top-0 bottom-0 w-px bg-white/20" />
                                            )}
                                        </div>
                                    </div>
                                )
                            ))}
                        </div>
                    ) : images.length === 1 ? (
                        <div
                            className="relative w-full h-full flex items-center justify-center"
                            ref={imageContainerRef}
                            onMouseDown={handleMouseDown}
                            onMouseMove={handleMouseMove}
                            onMouseUp={handleMouseUp}
                            onMouseLeave={handleMouseUp}
                            onWheel={handleWheel}
                            style={{ cursor: isPanMode ? (isPanning ? 'grabbing' : 'grab') : (activeTool && activeTool !== 'eraser' ? 'crosshair' : 'default') }}
                        >
                            <div
                                className="relative"
                                style={{
                                    transform: `translate(${position.x}px, ${position.y}px) rotate(${rotation}deg) scale(${zoom / 100})`,
                                    transition: isPanning ? 'none' : 'transform 300ms'
                                }}
                            >
                                <img
                                    ref={imageRef}
                                    src={images[0].url}
                                    alt={images[0].type}
                                    className="max-w-full max-h-full object-contain select-none"
                                    style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)` }}
                                    draggable={false}
                                />

                                {/* SVG overlay for annotations */}
                                {imageRef.current && (
                                    <svg
                                        className="absolute top-0 left-0 w-full h-full"
                                        viewBox={`0 0 ${imageRef.current.naturalWidth || imageRef.current.width} ${imageRef.current.naturalHeight || imageRef.current.height}`}
                                        preserveAspectRatio="none"
                                        style={{ overflow: 'visible' }}
                                    >
                                        {/* Render saved annotations */}
                                        {annotations.map((shape, idx) => (
                                            <g key={idx}>{renderShape(shape, idx, false)}</g>
                                        ))}
                                        {/* Render current drawing shape */}
                                        {currentShape && renderShape(currentShape, -1, true)}
                                        {/* Render saved measurements */}
                                        {measurements.map((measurement, idx) => 
                                            renderMeasurement(measurement, idx, false)
                                        )}
                                        {/* Render current measurement */}
                                        {currentMeasurement && renderMeasurement(currentMeasurement, -1, true)}
                                    </svg>
                                )}
                            </div>
                        </div>
                    ) : null}
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
                currentImage={images[0] || image}
                patientInfo={patientInfo}
                onCompareImages={(images, caseData) => handleCompareImages(images, caseData)}
            />

            {/* Delete Confirmation Modal */}
            <ConfirmModal
                isOpen={confirmDelete}
                onClose={handleCancelDelete}
                onConfirm={handleConfirmDelete}
                title="Xóa vùng khoanh"
                message="Bạn có chắc chắn muốn xóa vùng khoanh này không?"
                confirmText="Xóa"
                cancelText="Hủy"
                confirmColor="red"
            />
        </div>
    );
};