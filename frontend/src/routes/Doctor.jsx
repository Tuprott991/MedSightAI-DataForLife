import { useState, useEffect } from 'react';
import { Search, Filter, Users, Loader2 } from 'lucide-react';
import { getCachedPatientList, getPatients, searchPatients } from '../services/patientApi';
import { PatientCard } from '../components/custom/PatientCard';
import { Pagination } from '../components/custom/Pagination';
import { ITEMS_PER_PAGE } from '../constants/general';
import { useTranslation } from 'react-i18next';

export const Doctor = () => {
    const { t } = useTranslation();
    const [searchQuery, setSearchQuery] = useState('');
    const [currentPage, setCurrentPage] = useState(1);
    const normalizedSearchQuery = searchQuery.trim();
    const cachedList = getCachedPatientList(currentPage, ITEMS_PER_PAGE, normalizedSearchQuery);
    const [patients, setPatients] = useState([]);
    const [total, setTotal] = useState(0);
    const [isLoading, setIsLoading] = useState(() => !cachedList);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!cachedList) return;
        setPatients(cachedList.patients || []);
        setTotal(cachedList.total || 0);
        setIsLoading(false);
    }, [cachedList]);

    // Fetch patients data
    useEffect(() => {
        const fetchPatients = async () => {
            if (!cachedList) {
                setIsLoading(true);
            }
            setError(null);
            try {
                const data = normalizedSearchQuery
                    ? await searchPatients(normalizedSearchQuery, currentPage, ITEMS_PER_PAGE)
                    : await getPatients(currentPage, ITEMS_PER_PAGE);
                 
                setPatients(data.patients || []);
                setTotal(data.total || 0);
            } catch (err) {
                console.error('Error fetching patients:', err);
                setError(err.message);
            } finally {
                setIsLoading(false);
            }
        };

        fetchPatients();
    }, [cachedList, currentPage, normalizedSearchQuery]);

    // Calculate pagination
    const totalPages = Math.ceil(total / ITEMS_PER_PAGE);
    const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
    const endIndex = Math.min(startIndex + ITEMS_PER_PAGE, total);

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
                        <h1 className="text-3xl md:text-4xl font-bold">{t('doctor.title')}</h1>
                    </div>
                    <p className="text-gray-400 ml-13">
                        {t('doctor.searchPlaceholder')}
                    </p>
                </div>

                {/* Search and Filter Bar */}
                <div className="mb-8 flex flex-col md:flex-row gap-4">
                    {/* Search Input */}
                    <div className="flex-1 relative">
                        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                        <input
                            type="text"
                            placeholder={t('doctor.searchPlaceholder')}
                            value={searchQuery}
                            onChange={handleSearch}
                            className="w-full bg-white/5 border border-white/10 rounded-lg pl-12 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-teal-500 focus:bg-white/10 transition-all"
                        />
                    </div>

                    {/* Filter Button */}
                    <button className="flex items-center gap-2 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-500/50 px-6 py-3 rounded-lg transition-all cursor-pointer">
                        <Filter className="w-5 h-5" />
                        <span className="font-medium">{t('common.filter')}</span>
                    </button>
                </div>

                {/* Results Info */}
                <div className="mb-6 flex items-center justify-between">
                    <p className="text-gray-400">
                        {t('doctor.pagination.showing')} <span className="text-white font-semibold">{startIndex + 1}-{endIndex}</span> {t('doctor.pagination.of')} <span className="text-white font-semibold">{total}</span> {t('doctor.pagination.patients')}
                    </p>
                    {searchQuery && (
                        <button
                            onClick={() => setSearchQuery('')}
                            className="text-sm text-teal-400 hover:text-teal-300 transition-colors"
                        >
                            {t('common.clear')}
                        </button>
                    )}
                </div>

                {/* Loading State */}
                {isLoading ? (
                    <div className="flex items-center justify-center py-20">
                        <Loader2 className="w-12 h-12 text-teal-500 animate-spin" />
                    </div>
                ) : error ? (
                    <div className="text-center py-20">
                        <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                            <Users className="w-8 h-8 text-red-400" />
                        </div>
                        <h3 className="text-xl font-semibold mb-2 text-red-400">{t('common.error')}</h3>
                        <p className="text-gray-400">{error}</p>
                    </div>
                ) : patients.length > 0 ? (
                    <>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                            {patients.map((patient) => (
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
                        <h3 className="text-xl font-semibold mb-2">{t('doctor.noResults')}</h3>
                        <p className="text-gray-400">
                            {t('common.search')}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};
