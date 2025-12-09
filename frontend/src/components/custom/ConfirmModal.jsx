import { X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const ConfirmModal = ({
    isOpen,
    onClose,
    onConfirm,
    title,
    message,
    confirmText,
    cancelText,
    confirmColor = "red" // 'red', 'teal', 'amber'
}) => {
    const { t } = useTranslation();
    if (!isOpen) return null;

    // Use translation with fallback to props
    const modalTitle = title || t('common.confirm');
    const modalMessage = message || t('nav.logoutConfirm');
    const modalConfirmText = confirmText || t('common.confirm');
    const modalCancelText = cancelText || t('common.cancel');

    const getConfirmButtonClass = () => {
        switch (confirmColor) {
            case 'red':
                return 'bg-red-500 hover:bg-red-600 text-white';
            case 'teal':
                return 'bg-teal-500 hover:bg-teal-600 text-white';
            case 'amber':
                return 'bg-amber-500 hover:bg-amber-600 text-white';
            default:
                return 'bg-red-500 hover:bg-red-600 text-white';
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative bg-[#1a1a1a] border border-white/10 rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white">{modalTitle}</h3>
                    <button
                        onClick={onClose}
                        className="p-1 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Body */}
                <div className="px-6 py-6">
                    <p className="text-sm text-gray-300">{modalMessage}</p>
                </div>

                {/* Footer */}
                <div className="px-6 py-4 border-t border-white/10 flex items-center justify-end gap-3">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 hover:text-white rounded transition-colors"
                    >
                        {modalCancelText}
                    </button>
                    <button
                        onClick={() => {
                            onConfirm();
                            onClose();
                        }}
                        className={`px-4 py-2 text-sm rounded transition-colors ${getConfirmButtonClass()}`}
                    >
                        {modalConfirmText}
                    </button>
                </div>
            </div>
        </div>
    );
};
