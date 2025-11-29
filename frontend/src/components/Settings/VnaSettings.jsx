import { useState, useEffect } from 'react';
import { Database, Wifi, AlertCircle, CheckCircle, Loader2, Lock } from 'lucide-react';
import { mockVnaTest, mockSaveVnaConfig, loadVnaConfig } from '../../services/mockApi';
import { Toast } from '../custom/Toast';

/**
 * Component cài đặt VNA (Vendor Neutral Archive)
 * Cho phép cấu hình và test kết nối đến VNA server thông qua DICOMweb
 */
const VnaSettings = () => {
    // Form state
    const [config, setConfig] = useState({
        baseUrl: '',
        qidoEndpoint: '/qido-rs',
        wadoEndpoint: '/wado-rs',
        stowEndpoint: '/stow-rs',
        authType: 'none', // none, basic, token
        username: '',
        password: '',
        token: ''
    });

    // UI state
    const [isTesting, setIsTesting] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [testResult, setTestResult] = useState(null);
    const [toast, setToast] = useState(null);
    const [showPassword, setShowPassword] = useState(false);

    // Load saved config khi mount
    useEffect(() => {
        const savedConfig = loadVnaConfig();
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
            const result = await mockVnaTest(config);
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
            const result = await mockSaveVnaConfig(config);
            setToast({
                type: 'success',
                message: result.message
            });
        } catch (error) {
            setToast({
                type: 'error',
                message: 'Lỗi khi lưu cấu hình'
            });
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div className="bg-[#141414] rounded-xl border border-white/10 p-6">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="p-3 bg-teal-500/10 rounded-lg">
                    <Database className="w-6 h-6 text-teal-500" />
                </div>
                <div>
                    <h2 className="text-xl font-bold text-white">Cài đặt VNA</h2>
                    <p className="text-sm text-gray-400">Cấu hình kết nối đến VNA Server (DICOMweb)</p>
                </div>
            </div>

            {/* Form */}
            <div className="space-y-4">
                {/* Base URL */}
                <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                        Base URL <span className="text-red-400">*</span>
                    </label>
                    <input
                        type="text"
                        value={config.baseUrl}
                        onChange={(e) => handleChange('baseUrl', e.target.value)}
                        placeholder="https://vna.example.com hoặc http://192.168.1.200:8080"
                        className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                    />
                </div>

                {/* DICOMweb Endpoints */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            QIDO Endpoint
                        </label>
                        <input
                            type="text"
                            value={config.qidoEndpoint}
                            onChange={(e) => handleChange('qidoEndpoint', e.target.value)}
                            placeholder="/qido-rs"
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                        />
                        <p className="text-xs text-gray-500 mt-1">Query (QIDO-RS)</p>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            WADO Endpoint
                        </label>
                        <input
                            type="text"
                            value={config.wadoEndpoint}
                            onChange={(e) => handleChange('wadoEndpoint', e.target.value)}
                            placeholder="/wado-rs"
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                        />
                        <p className="text-xs text-gray-500 mt-1">Retrieve (WADO-RS)</p>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            STOW Endpoint
                        </label>
                        <input
                            type="text"
                            value={config.stowEndpoint}
                            onChange={(e) => handleChange('stowEndpoint', e.target.value)}
                            placeholder="/stow-rs"
                            className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                        />
                        <p className="text-xs text-gray-500 mt-1">Store (STOW-RS)</p>
                    </div>
                </div>

                {/* Authentication Type */}
                <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                        Phương thức xác thực
                    </label>
                    <select
                        value={config.authType}
                        onChange={(e) => handleChange('authType', e.target.value)}
                        className="w-full px-4 py-2.5 bg-[#1a1a1a] border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                    >
                        <option value="none">Không xác thực</option>
                        <option value="basic">Basic Authentication</option>
                        <option value="token">Token/Bearer</option>
                    </select>
                </div>

                {/* Basic Auth Fields */}
                {config.authType === 'basic' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-[#1a1a1a] rounded-lg border border-white/5">
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Username <span className="text-red-400">*</span>
                            </label>
                            <input
                                type="text"
                                value={config.username}
                                onChange={(e) => handleChange('username', e.target.value)}
                                placeholder="admin"
                                className="w-full px-4 py-2.5 bg-[#0a0a0a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Password <span className="text-red-400">*</span>
                            </label>
                            <div className="relative">
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    value={config.password}
                                    onChange={(e) => handleChange('password', e.target.value)}
                                    placeholder="••••••••"
                                    className="w-full px-4 py-2.5 pr-10 bg-[#0a0a0a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all"
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
                                >
                                    <Lock className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Token Auth Field */}
                {config.authType === 'token' && (
                    <div className="p-4 bg-[#1a1a1a] rounded-lg border border-white/5">
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                            Access Token <span className="text-red-400">*</span>
                        </label>
                        <textarea
                            value={config.token}
                            onChange={(e) => handleChange('token', e.target.value)}
                            placeholder="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                            rows="3"
                            className="w-full px-4 py-2.5 bg-[#0a0a0a] border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 transition-all font-mono text-sm"
                        />
                        <p className="text-xs text-gray-500 mt-1">JWT token hoặc API key</p>
                    </div>
                )}
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
                                    <p>• Server Version: {testResult.details.serverVersion}</p>
                                    <p>• Response Time: {testResult.details.responseTime}</p>
                                    <p>• Capabilities: {testResult.details.capabilities.join(', ')}</p>
                                    <p>• Auth Type: {testResult.details.authType}</p>
                                </div>
                            )}
                            {!testResult.success && testResult.errorCode && (
                                <p className="mt-1 text-xs text-gray-500">Error Code: {testResult.errorCode}</p>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3 mt-6">
                <button
                    onClick={handleTest}
                    disabled={isTesting || !config.baseUrl}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-teal-500 hover:bg-teal-600 disabled:bg-teal-500/50 text-white font-medium rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2 focus:ring-offset-[#141414] disabled:cursor-not-allowed"
                >
                    {isTesting ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>Đang kiểm tra...</span>
                        </>
                    ) : (
                        <>
                            <Wifi className="w-4 h-4" />
                            <span>Test Connection</span>
                        </>
                    )}
                </button>

                <button
                    onClick={handleSave}
                    disabled={isSaving || !config.baseUrl}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-500 hover:bg-blue-600 disabled:bg-blue-500/50 text-white font-medium rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-[#141414] disabled:cursor-not-allowed"
                >
                    {isSaving ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>Đang lưu...</span>
                        </>
                    ) : (
                        <span>Lưu cấu hình</span>
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

export default VnaSettings;
