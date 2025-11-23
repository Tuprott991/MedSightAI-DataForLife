import { useEffect } from 'react';
import { CheckCircle2, AlertCircle, X } from 'lucide-react';

export const Toast = ({ type, message, onClose }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onClose();
        }, 3000);

        return () => clearTimeout(timer);
    }, [onClose]);

    return (
        <div className="fixed top-6 right-6 z-50 animate-slide-in-right">
            <div className={`flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg backdrop-blur-sm border min-w-[300px] max-w-md ${type === 'success'
                    ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                    : 'bg-red-500/20 text-red-400 border-red-500/30'
                }`}>
                {type === 'success' ? (
                    <CheckCircle2 className="w-5 h-5 shrink-0" />
                ) : (
                    <AlertCircle className="w-5 h-5 shrink-0" />
                )}
                <span className="flex-1 text-sm font-medium">{message}</span>
                <button
                    onClick={onClose}
                    className="shrink-0 hover:bg-white/10 rounded p-1 transition-colors"
                >
                    <X className="w-4 h-4" />
                </button>
            </div>
        </div>
    );
};
