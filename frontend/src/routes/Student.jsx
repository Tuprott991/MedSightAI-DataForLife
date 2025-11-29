import { useState, useMemo } from 'react';
import { Search, Filter, Users } from 'lucide-react';
import { patientsData } from '../constants/patients';
import { PatientCard } from '../components/custom/PatientCard';
import { Pagination } from '../components/custom/Pagination';
import { ITEMS_PER_PAGE } from '../constants/general';

export const Student = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [currentPage, setCurrentPage] = useState(1);

    // Filter patients based on search query
    const filteredPatients = useMemo(() => {
        return patientsData.filter(patient =>
            patient.name.toLowerCase().includes(searchQuery.toLowerCase())
        );
    }, [searchQuery]);

    // Calculate pagination
    const totalPages = Math.ceil(filteredPatients.length / ITEMS_PER_PAGE);
    const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
    const endIndex = startIndex + ITEMS_PER_PAGE;
    const currentPatients = filteredPatients.slice(startIndex, endIndex);

    // Handle search
    const handleSearch = (e) => {
        setSearchQuery(e.target.value);
        setCurrentPage(1); // Reset to first page when searching
    };

    // Handle page change
    const handlePageChange = (page) => {
        setCurrentPage(page);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    return (
        <div className="min-h-screen bg-[#1b1b1b] text-white">
            <div className="container mx-auto px-6 py-8">
                {/* Header */}
                <div className="mb-8">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="w-10 h-10 bg-teal-500/20 rounded-lg flex items-center justify-center">
                            <Users className="w-6 h-6 text-teal-500" />
                        </div>
                        <h1 className="text-3xl md:text-4xl font-bold">Hồ Sơ Bệnh Nhân</h1>
                    </div>
                    <p className="text-gray-400 ml-13">
                        Quản lý và theo dõi hồ sơ bệnh án
                    </p>
                </div>

                {/* Search and Filter Bar */}
                <div className="mb-8 flex flex-col md:flex-row gap-4">
                    {/* Search Input */}
                    <div className="flex-1 relative">
                        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                        <input
                            type="text"
                            placeholder="Tìm kiếm theo tên bệnh nhân..."
                            value={searchQuery}
                            onChange={handleSearch}
                            className="w-full bg-white/5 border border-white/10 rounded-lg pl-12 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-teal-500 focus:bg-white/10 transition-all"
                        />
                    </div>

                    {/* Filter Button */}
                    <button className="flex items-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-500/50 px-6 py-3 rounded-lg transition-all cursor-pointer">
                        <Filter className="w-5 h-5" />
                        <span className="font-medium">Lọc</span>
                    </button>
                </div>

                {/* Results Info */}
                <div className="mb-6 flex items-center justify-between">
                    <p className="text-gray-400">
                        Hiển thị <span className="text-white font-semibold">{startIndex + 1}-{Math.min(endIndex, filteredPatients.length)}</span> trong tổng số <span className="text-white font-semibold">{filteredPatients.length}</span> bệnh nhân
                    </p>
                    {searchQuery && (
                        <button
                            onClick={() => setSearchQuery('')}
                            className="text-sm text-teal-400 hover:text-teal-300 transition-colors"
                        >
                            Xóa tìm kiếm
                        </button>
                    )}
                </div>

                {/* Patient Grid */}
                {currentPatients.length > 0 ? (
                    <>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                            {currentPatients.map((patient) => (
                                <PatientCard key={patient.id} patient={patient} />
                            ))}
                        </div>

                        {/* Pagination */}
                        {totalPages > 1 && (
                            <Pagination
                                currentPage={currentPage}
                                totalPages={totalPages}
                                onPageChange={handlePageChange}
                            />
                        )}
                    </>
                ) : (
                    <div className="text-center py-20">
                        <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-4">
                            <Search className="w-8 h-8 text-gray-400" />
                        </div>
                        <h3 className="text-xl font-semibold mb-2">Không tìm thấy bệnh nhân</h3>
                        <p className="text-gray-400">
                            Thử điều chỉnh từ khóa tìm kiếm
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};