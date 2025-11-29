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
            </div>
        </div>
    );
};
