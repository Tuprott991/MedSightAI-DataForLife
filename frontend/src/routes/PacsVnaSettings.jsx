import { useState } from 'react';
import PacsSettings from '../components/Settings/PacsSettings';
import VnaSettings from '../components/Settings/VnaSettings';

/**
 * Trang cài đặt PACS/VNA
 * Cho phép người dùng cấu hình kết nối đến PACS và VNA servers
 */
export const PacsVnaSettings = () => {
    const [activeTab, setActiveTab] = useState('pacs'); // 'pacs' hoặc 'vna'

    return (
        <div className="min-h-screen bg-[#1b1b1b] p-6">
            <div className="max-w-4xl mx-auto">
                {/* Page Header */}
                <div className="mb-6">
                    <h1 className="text-3xl font-bold text-white mb-2">Cài đặt PACS/VNA</h1>
                    <p className="text-gray-400">
                        Cấu hình kết nối đến hệ thống lưu trữ hình ảnh y tế (PACS) và kho lưu trữ trung lập (VNA)
                    </p>
                </div>

                {/* Tabs */}
                <div className="flex gap-2 mb-6">
                    <button
                        onClick={() => setActiveTab('pacs')}
                        className={`flex-1 px-6 py-3 font-medium rounded-lg transition-all ${activeTab === 'pacs'
                            ? 'bg-teal-500 text-white shadow-lg shadow-teal-500/50'
                            : 'bg-[#141414] text-gray-400 hover:text-white hover:bg-white/5'
                            }`}
                    >
                        PACS Settings
                    </button>
                    <button
                        onClick={() => setActiveTab('vna')}
                        className={`flex-1 px-6 py-3 font-medium rounded-lg transition-all ${activeTab === 'vna'
                            ? 'bg-teal-500 text-white shadow-lg shadow-teal-500/50'
                            : 'bg-[#141414] text-gray-400 hover:text-white hover:bg-white/5'
                            }`}
                    >
                        VNA Settings
                    </button>
                </div>

                {/* Tab Content */}
                <div className="transition-all duration-300">
                    {activeTab === 'pacs' && <PacsSettings />}
                    {activeTab === 'vna' && <VnaSettings />}
                </div>

                {/* Info Section */}
                <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                    <h3 className="text-sm font-semibold text-blue-400 mb-2">ℹ️ Thông tin</h3>
                    <ul className="text-sm text-gray-400 space-y-1">
                        <li>• <strong>PACS (Picture Archiving and Communication System):</strong> Hệ thống lưu trữ và truyền thông hình ảnh y tế sử dụng giao thức DICOM.</li>
                        <li>• <strong>VNA (Vendor Neutral Archive):</strong> Kho lưu trữ trung lập sử dụng chuẩn DICOMweb (QIDO-RS, WADO-RS, STOW-RS).</li>
                        <li>• Tất cả cấu hình được lưu trữ cục bộ trên trình duyệt của bạn.</li>
                        <li>• Sử dụng chức năng "Test Connection" để kiểm tra kết nối trước khi lưu.</li>
                    </ul>
                </div>
            </div>
        </div>
    );
};
