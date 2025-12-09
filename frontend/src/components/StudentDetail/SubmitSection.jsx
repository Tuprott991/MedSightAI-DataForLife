import { useState } from 'react';
import { Send } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const SubmitSection = ({ onSubmit, annotations = [], showToast }) => {
    const { t } = useTranslation();
    const [diagnosis, setDiagnosis] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleSubmit = () => {
        if (!diagnosis.trim()) {
            showToast?.('error', t('studentDetail.submit.error'));
            return;
        }

        if (isSubmitting) return;
        setIsSubmitting(true);

        const submissionData = {
            diagnosis,
            annotations,
            timestamp: new Date().toISOString()
        };

        try {
            const result = onSubmit?.(submissionData);
            if (result?.success !== false) {
                showToast?.('success', t('studentDetail.submit.success'));
                // Không clear diagnosis và annotations để user có thể xem lại
                // setDiagnosis('');
            } else {
                showToast?.('error', t('studentDetail.submit.error'));
            }
        } catch (error) {
            showToast?.('error', t('studentDetail.submit.error'));
        } finally {
            setTimeout(() => setIsSubmitting(false), 500);
        }
    };

    return (
        <div className="h-24 bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden flex flex-col shrink-0">
            {/* Form Content */}
            <div className="flex-1 p-4">
                <div className="flex items-end justify-between gap-3">
                    {/* Diagnosis Input */}
                    <div className="flex-1">
                        <label className="text-xs text-gray-400 mb-1 block">
                            {t('studentDetail.submit.diagnosis')} <span className="text-red-400">*</span>
                        </label>
                        <input
                            type="text"
                            value={diagnosis}
                            onChange={(e) => setDiagnosis(e.target.value)}
                            placeholder={t('studentDetail.submit.diagnosisPlaceholder')}
                            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-teal-500 transition-colors"
                        />
                    </div>

                    {/* Submit Button */}
                    <button
                        onClick={handleSubmit}
                        disabled={isSubmitting}
                        className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all shrink-0 bg-teal-500 hover:bg-teal-600 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <Send className="w-4 h-4" />
                        <span>{isSubmitting ? t('common.loading') : t('studentDetail.submit.submit')}</span>
                    </button>
                </div>
            </div>
        </div>
    );
};
