import { useEffect, useState } from 'react';
import { Loader2, Image as ImageIcon } from 'lucide-react';

const loadedImageUrls = new Set();

/**
 * Component to display medical images (PNG/JPG) from S3 URL.
 */
export const DicomImage = ({ src, alt, className, onLoad, onError }) => {
    const [isLoading, setIsLoading] = useState(() => Boolean(src) && !loadedImageUrls.has(src));
    const [error, setError] = useState(false);

    useEffect(() => {
        setError(false);
        setIsLoading(Boolean(src) && !loadedImageUrls.has(src));
    }, [src]);

    const handleLoad = () => {
        if (src) {
            loadedImageUrls.add(src);
        }
        setIsLoading(false);
        setError(false);
        if (onLoad) onLoad();
    };

    const handleError = (event) => {
        setIsLoading(false);
        setError(true);
        if (onError) onError(event);
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
                style={{ display: error ? 'none' : 'block' }}
            />
        </div>
    );
};
