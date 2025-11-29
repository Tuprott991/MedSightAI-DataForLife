import { Activity } from 'lucide-react';

export const ImageList = ({ images, selectedImage, onImageSelect }) => {
    return (
        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-4">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-teal-500" />
                Medical Images
            </h3>
            <div className="space-y-3 max-h-[800px] overflow-y-auto">
                {images.map((image) => (
                    <div
                        key={image.id}
                        onClick={() => onImageSelect(image)}
                        className={`group cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${selectedImage.id === image.id
                            ? 'border-teal-500 shadow-lg shadow-teal-500/30'
                            : 'border-white/10 hover:border-teal-500/50'
                            }`}
                    >
                        <div className="relative h-32">
                            <img
                                src={image.url}
                                alt={image.type}
                                className="w-full h-full object-cover group-hover:scale-110 transition-transform"
                            />
                            <div className="absolute inset-0 bg-linear-to-t from-black/80 to-transparent"></div>
                            <div className="absolute bottom-2 left-2 right-2">
                                <p className="text-xs font-semibold text-white">{image.type}</p>
                                <div className="flex items-center justify-between mt-1">
                                    <p className="text-xs text-gray-300">{image.modality}</p>
                                    <p className="text-xs text-gray-400">{new Date(image.date).toLocaleDateString()}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
