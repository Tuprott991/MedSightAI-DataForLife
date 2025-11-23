import { useState } from 'react';
import { Send } from 'lucide-react';

export const SubmitSection = ({ onSubmit, annotations = [], showToast }) => {
    const [diagnosis, setDiagnosis] = useState('');

    const handleSubmit = () => {
        if (!diagnosis.trim()) {
            showToast?.('error', 'Please enter a diagnosis');
            return;
        }

        const submissionData = {
            diagnosis,
            annotations,
            timestamp: new Date().toISOString()
        };

        try {
            const result = onSubmit?.(submissionData);
            if (result?.success !== false) {
                showToast?.('success', 'Diagnosis submitted successfully!');
                setDiagnosis('');
            } else {
                showToast?.('error', 'Failed to submit diagnosis. Please try again.');
            }
        } catch (error) {
            showToast?.('error', 'Failed to submit diagnosis. Please try again.');
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
                            Diagnosis <span className="text-red-400">*</span>
                        </label>
                        <input
                            type="text"
                            value={diagnosis}
                            onChange={(e) => setDiagnosis(e.target.value)}
                            placeholder="e.g., Coronary Artery Disease"
                            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-teal-500 transition-colors"
                        />
                    </div>

                    {/* Submit Button */}
                    <button
                        onClick={handleSubmit}
                        className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all shrink-0 bg-teal-500 hover:bg-teal-600 text-white"
                    >
                        <Send className="w-4 h-4" />
                        <span>Submit</span>
                    </button>
                </div>
            </div>
        </div>
    );
};
