import { useState, useEffect } from 'react';
import { Server, Wifi, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { mockPacsTest, mockSavePacsConfig, loadPacsConfig } from '../../services/mockApi';
import { importPatientCase } from '../../services/patientApi';
import { Toast } from '../custom/Toast';

/**
 * Component cài đặt PACS
 * Cho phép cấu hình và test kết nối đến PACS server
 */
const PacsSettings = () => {
    const { t } = useTranslation();
    // Form state
    const [config, setConfig] = useState({
        host: '',
        port: 104,
        localAETitle: 'MEDSIGHT',
        remoteAETitle: 'PACS'
    });

    // UI state
    const [isTesting, setIsTesting] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [isImporting, setIsImporting] = useState(false);
    const [testResult, setTestResult] = useState(null);
    const [toast, setToast] = useState(null);
    const [importForm, setImportForm] = useState({
        name: '',
        age: '',
        gender: 'Male',
        phoneNumber: '',
        diagnosis: '',
        findings: '',
        imageFile: null,
    });

    // Load saved config khi mount
    useEffect(() => {
        const savedConfig = loadPacsConfig();
        if (savedConfig) {
            setConfig(savedConfig);
        }
    }, []);

    // Handle input change
    const handleChange = (field, value) => {
        setConfig(prev => ({
            ...prev,
            [field]: value
        }));
        // Clear test result khi thay đổi config
        setTestResult(null);
    };

    // Test connection
    const handleTest = async () => {
        setIsTesting(true);
        setTestResult(null);

        try {
            const result = await mockPacsTest(config);
            setTestResult(result);
            setToast({
                type: 'success',
                message: result.message
            });
        } catch (error) {
            setTestResult(error);
            setToast({
                type: 'error',
                message: error.message
            });
        } finally {
            setIsTesting(false);
        }
    };

    // Save configuration
    const handleSave = async () => {
        setIsSaving(true);

        try {
            const result = await mockSavePacsConfig(config);
            setToast({
                type: 'success',
                message: result.message
            });
        } catch (error) {
            setToast({
                type: 'error',
                message: t('settings.error')
            });
        } finally {
            setIsSaving(false);
        }
    };

    const handleImportChange = (field, value) => {
        setImportForm((previous) => ({
            ...previous,
            [field]: value,
        }));
    };

    const handleImportCase = async () => {
        setIsImporting(true);

        try {
            await importPatientCase({
                name: importForm.name.trim(),
                age: importForm.age ? Number(importForm.age) : null,
                gender: importForm.gender,
                phoneNumber: importForm.phoneNumber.trim() || null,
                diagnosis: importForm.diagnosis.trim() || null,
                findings: importForm.findings.trim() || null,
                file: importForm.imageFile,
            });

            setImportForm({
                name: '',
                age: '',
                gender: 'Male',
                phoneNumber: '',
                diagnosis: '',
                findings: '',
                imageFile: null,
            });

            setToast({
                type: 'success',
                message: t('settings.pacs.importSuccess'),
            });
        } catch (error) {
            setToast({
                type: 'error',
                message: error.message || t('settings.error'),
            });
        } finally {
            setIsImporting(false);
        }
    };

    return (
        <div className="bg-[#141414] rounded-xl border border-white/10 p-6">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="p-3 bg-teal-500/10 rounded-lg">
                    <Server className="w-6 h-6 text-teal-500" />
                </div>
                <div>
                    <h2 className="text-xl font-bold text-white">{t('settings.pacs.title')}</h2>
                    <p className="text-sm text-gray-400">{t('settings.pacs.subtitle')}</p>
                </div>
            </div>

            {/* Form */}
            <div className="space-y-4">
                {/* Host */}
                <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                        {t('settings.pacs.host')} <span className="text-red-400">{t('settings.required')}</span>
                    </label>
                    <input
                        type="text"
                        value={config.host}
                        onChange={(e) => handleChange('host', e.target.value)}
                        placeholder={t('settings.pacs.hostPlaceholder')}
                        className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                    />
                </div>

                {/* Port */}
                <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                        {t('settings.pacs.port')} <span className="text-red-400">{t('settings.required')}</span>
                    </label>
                    <input
                        type="number"
                        value={config.port}
                        onChange={(e) => handleChange('port', parseInt(e.target.value) || 0)}
                        placeholder={t('settings.pacs.portPlaceholder')}
                        min="1"
                        max="65535"
                        className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                    />
                    <p className="text-xs text-gray-500 mt-1">{t('settings.pacs.portHelp')}</p>
                </div>

                {/* Local AE Title */}
                <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                        {t('settings.pacs.localAETitle')} <span className="text-red-400">{t('settings.required')}</span>
                    </label>
                    <input
                        type="text"
                        value={config.localAETitle}
                        onChange={(e) => handleChange('localAETitle', e.target.value.toUpperCase())}
                        placeholder={t('settings.pacs.localAETitlePlaceholder')}
                        maxLength="16"
                        className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                    />
                    <p className="text-xs text-gray-500 mt-1">{t('settings.pacs.localAETitleHelp')}</p>
                </div>

                {/* Remote AE Title */}
                <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                        {t('settings.pacs.remoteAETitle')} <span className="text-red-400">{t('settings.required')}</span>
                    </label>
                    <input
                        type="text"
                        value={config.remoteAETitle}
                        onChange={(e) => handleChange('remoteAETitle', e.target.value.toUpperCase())}
                        placeholder={t('settings.pacs.remoteAETitlePlaceholder')}
                        maxLength="16"
                        className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                    />
                    <p className="text-xs text-gray-500 mt-1">{t('settings.pacs.remoteAETitleHelp')}</p>
                </div>
            </div>

            {/* Test Result */}
            {testResult && (
                <div className={`mt-6 p-4 rounded-lg border ${testResult.success
                    ? 'bg-green-500/10 border-green-500/20'
                    : 'bg-red-500/10 border-red-500/20'
                    }`}>
                    <div className="flex items-start gap-3">
                        {testResult.success ? (
                            <CheckCircle className="w-5 h-5 text-green-500 shrink-0 mt-0.5" />
                        ) : (
                            <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                        )}
                        <div className="flex-1">
                            <p className={`font-medium ${testResult.success ? 'text-green-400' : 'text-red-400'
                                }`}>
                                {testResult.message}
                            </p>
                            {testResult.success && testResult.details && (
                                <div className="mt-2 text-sm text-gray-400 space-y-1">
                                    <p>• {t('settings.pacs.serverInfo')}: {testResult.details.serverInfo}</p>
                                    <p>• {t('settings.pacs.responseTime')}: {testResult.details.responseTime}</p>
                                    <p>• {t('settings.pacs.localAE')}: {testResult.details.localAE}</p>
                                    <p>• {t('settings.pacs.remoteAE')}: {testResult.details.remoteAE}</p>
                                </div>
                            )}
                            {!testResult.success && testResult.errorCode && (
                                <p className="mt-1 text-xs text-gray-500">{t('settings.pacs.errorCode')}: {testResult.errorCode}</p>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3 mt-6">
                <button
                    onClick={handleTest}
                    disabled={isTesting || !config.host || !config.port}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-teal-500 hover:bg-teal-600 disabled:bg-teal-500/50 text-white font-medium rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2 focus:ring-offset-[#141414] disabled:cursor-not-allowed"
                >
                    {isTesting ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>{t('settings.pacs.testing')}</span>
                        </>
                    ) : (
                        <>
                            <Wifi className="w-4 h-4" />
                            <span>{t('settings.pacs.testConnection')}</span>
                        </>
                    )}
                </button>

                <button
                    onClick={handleSave}
                    disabled={isSaving || !config.host || !config.port}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-500 hover:bg-blue-600 disabled:bg-blue-500/50 text-white font-medium rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-[#141414] disabled:cursor-not-allowed"
                >
                    {isSaving ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>{t('settings.pacs.saving')}</span>
                        </>
                    ) : (
                        <span>{t('settings.pacs.save')}</span>
                    )}
                </button>
            </div>

            <div className="mt-8 pt-8 border-t border-white/10">
                <div className="mb-6">
                    <h3 className="text-lg font-semibold text-white">{t('settings.pacs.importTitle')}</h3>
                    <p className="text-sm text-gray-400 mt-1">{t('settings.pacs.importSubtitle')}</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            {t('settings.pacs.patientName')} <span className="text-red-400">{t('settings.required')}</span>
                        </label>
                        <input
                            type="text"
                            value={importForm.name}
                            onChange={(e) => handleImportChange('name', e.target.value)}
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            {t('settings.pacs.patientAge')}
                        </label>
                        <input
                            type="number"
                            min="0"
                            value={importForm.age}
                            onChange={(e) => handleImportChange('age', e.target.value)}
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            {t('settings.pacs.patientGender')}
                        </label>
                        <select
                            value={importForm.gender}
                            onChange={(e) => handleImportChange('gender', e.target.value)}
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                        >
                            <option value="Male">{t('doctorDetail.patientInfo.male')}</option>
                            <option value="Female">{t('doctorDetail.patientInfo.female')}</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            {t('settings.pacs.patientPhone')}
                        </label>
                        <input
                            type="text"
                            value={importForm.phoneNumber}
                            onChange={(e) => handleImportChange('phoneNumber', e.target.value)}
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                        />
                    </div>
                </div>

                <div className="mt-4">
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                        {t('settings.pacs.imageFile')} <span className="text-red-400">{t('settings.required')}</span>
                    </label>
                    <input
                        type="file"
                        accept=".png,.jpg,.jpeg,.dcm,.dicom,image/png,image/jpeg,application/dicom"
                        onChange={(e) => handleImportChange('imageFile', e.target.files?.[0] || null)}
                        className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white file:mr-4 file:px-4 file:py-2 file:border-0 file:rounded-md file:bg-teal-500/20 file:text-teal-300 hover:file:bg-teal-500/30 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                    />
                    {importForm.imageFile && (
                        <p className="text-xs text-gray-400 mt-2">{importForm.imageFile.name}</p>
                    )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            {t('settings.pacs.caseDiagnosis')}
                        </label>
                        <input
                            type="text"
                            value={importForm.diagnosis}
                            onChange={(e) => handleImportChange('diagnosis', e.target.value)}
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            {t('settings.pacs.caseFindings')}
                        </label>
                        <input
                            type="text"
                            value={importForm.findings}
                            onChange={(e) => handleImportChange('findings', e.target.value)}
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                        />
                    </div>
                </div>

                <button
                    onClick={handleImportCase}
                    disabled={isImporting || !importForm.name.trim() || !importForm.imageFile}
                    className="mt-6 w-full md:w-auto flex items-center justify-center gap-2 px-5 py-3 bg-emerald-500 hover:bg-emerald-600 disabled:bg-emerald-500/40 text-white font-medium rounded-lg transition-all disabled:cursor-not-allowed"
                >
                    {isImporting ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>{t('settings.pacs.importing')}</span>
                        </>
                    ) : (
                        <span>{t('settings.pacs.importAction')}</span>
                    )}
                </button>
            </div>

            {/* Toast Notification */}
            {toast && (
                <Toast
                    type={toast.type}
                    message={toast.message}
                    onClose={() => setToast(null)}
                />
            )}
        </div>
    );
};

export default PacsSettings;
