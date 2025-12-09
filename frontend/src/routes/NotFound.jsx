import { Link } from 'react-router-dom';
import { Home, ArrowLeft, AlertCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const NotFound = () => {
    const { t } = useTranslation();

    return (
        <div className="min-h-screen bg-[#1b1b1b] text-white flex items-center justify-center px-6">
            <div className="max-w-2xl w-full text-center">
                {/* 404 Number with Animation */}
                <div className="relative mb-8">
                    <h1 className="text-[150px] md:text-[200px] font-bold bg-linear-to-r from-teal-500 to-teal-300 bg-clip-text text-transparent leading-none">
                        404
                    </h1>
                    <div className="absolute inset-0 bg-linear-to-r from-teal-500/20 to-teal-300/20 blur-3xl"></div>
                </div>

                {/* Error Icon */}
                <div className="flex justify-center mb-6">
                    <div className="w-20 h-20 bg-teal-500/20 rounded-full flex items-center justify-center border border-teal-500/30">
                        <AlertCircle className="w-10 h-10 text-teal-500" />
                    </div>
                </div>

                {/* Error Message */}
                <h2 className="text-3xl md:text-4xl font-bold mb-4">{t('notFound.title')}</h2>
                <p className="text-gray-400 text-lg mb-8 max-w-md mx-auto">
                    {t('notFound.description')}
                </p>

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Link
                        to="/home"
                        className="group inline-flex items-center justify-center gap-2 bg-teal-500 hover:bg-teal-600 text-white px-8 py-4 rounded-lg font-semibold transition-all transform hover:scale-105 shadow-lg shadow-teal-500/50"
                    >
                        <Home className="w-5 h-5" />
                        {t('notFound.backHome')}
                        <span className="absolute inset-0 rounded-lg bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity"></span>
                    </Link>

                    <button
                        onClick={() => window.history.back()}
                        className="inline-flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 text-white px-8 py-4 rounded-lg font-semibold transition-all transform hover:scale-105 border border-white/20"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        {t('notFound.goBack')}
                    </button>
                </div>

                {/* Helpful Links */}
                <div className="mt-12 pt-8 border-t border-white/10">
                    <p className="text-gray-400 mb-4">{t('notFound.helpfulLinks')}</p>
                    <div className="flex flex-wrap gap-3 justify-center">
                        <Link
                            to="/doctor"
                            className="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-500/50 rounded-lg text-sm transition-all"
                        >
                            {t('notFound.doctorPortal')}
                        </Link>
                        <Link
                            to="/student"
                            className="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-teal-500/50 rounded-lg text-sm transition-all"
                        >
                            {t('notFound.studentPortal')}
                        </Link>
                    </div>
                </div>

                {/* Decorative Elements */}
                <div className="absolute inset-0 overflow-hidden pointer-events-none">
                    <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-teal-500/5 rounded-full blur-3xl"></div>
                    <div className="absolute bottom-1/4 right-1/4 w-64 h-64 bg-teal-500/5 rounded-full blur-3xl"></div>
                </div>
            </div>
        </div>
    );
};