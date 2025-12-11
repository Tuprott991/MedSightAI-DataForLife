import { useState } from 'react';
import { Loader2, Image as ImageIcon } from 'lucide-react';

/**
 * Component to display medical images (PNG/JPG) from S3 URL
 */
export const DicomImage = ({ src, alt, className, onLoad, onError }) => {
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(false);

    const handleLoad = () => {
        setIsLoading(false);
        setError(false);
        console.log('✅ Image loaded successfully:', src);
        if (onLoad) onLoad();
    };

    const handleError = (e) => {
        setIsLoading(false);
        setError(true);
        console.error('❌ Image failed to load:', src, e);
        if (onError) onError(e);
    };

    if (!src) {
        return (
            <div className={`flex items-center justify-center bg-gray-800/50 rounded-lg ${className}`}>
                <div className="text-center">
                    <ImageIcon className="w-12 h-12 text-gray-600 mx-auto mb-2" />
                    <p className="text-gray-400 text-sm">No image available</p>
                </div>
            </div>
        );
    }

    console.log('🖼️  Loading image from:', src);

    return (
        <div className={`relative ${className || ''}`}>
            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-800/50 rounded-lg z-10">
                    <Loader2 className="w-8 h-8 text-teal-500 animate-spin" />
                </div>
            )}
            
            {error && !isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-800/50 rounded-lg">
                    <div className="text-center">
                        <ImageIcon className="w-12 h-12 text-red-600 mx-auto mb-2" />
                        <p className="text-red-400 text-sm">Failed to load image</p>
                        <p className="text-red-300 text-xs mt-1 px-2 break-all">{src}</p>
                    </div>
                </div>
            )}
            
            <img
                src={src}
                alt={alt || 'Medical Image'}
                loading="lazy"
                decoding="async"
                className={`${className || ''} ${isLoading ? 'opacity-0' : 'opacity-100'} transition-opacity duration-300`}
                onLoad={handleLoad}
                onError={handleError}
                crossOrigin="anonymous"
                style={{ display: error ? 'none' : 'block' }}
            />
        </div>
    );
};
