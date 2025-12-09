import { Link } from 'react-router-dom';
import { Stethoscope, GraduationCap, Activity, Sparkles, TrendingUp, Users } from 'lucide-react';
import { AnimatedCounter } from './AnimatedCounter';
import { useTranslation } from 'react-i18next';

export const HeroSection = () => {
    const { t } = useTranslation();

    return (
        <div className="relative overflow-hidden">
            {/* Background gradient */}
            <div className="absolute inset-0 bg-linear-to-br from-teal-500/20 to-transparent"></div>

            {/* Animated background elements */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-20 left-10 w-72 h-72 bg-teal-500/10 rounded-full blur-3xl animate-pulse"></div>
                <div className="absolute bottom-20 right-10 w-96 h-96 bg-teal-600/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
            </div>

            <div className="container mx-auto px-6 py-24 md:py-32 relative z-10">
                <div className="max-w-5xl mx-auto">
                    {/* Badge */}
                    <div className="flex justify-center mb-8 animate-fade-in">
                        <div className="inline-flex items-center gap-2 bg-teal-500/20 border border-teal-500/30 rounded-full px-6 py-2">
                            <Activity className="w-4 h-4 text-teal-500" />
                            <span className="text-sm text-teal-400 font-medium">{t('home.hero.badge')}</span>
                        </div>
                    </div>

                    {/* Main heading */}
                    <div className="text-center mb-6">
                        <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold mb-4 bg-linear-to-r from-white via-teal-200 to-teal-500 bg-clip-text text-transparent pb-4">
                            {t('app.name')}
                        </h1>
                    </div>

                    {/* Description */}
                    <p
                        className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto mb-8 text-center leading-relaxed"
                        dangerouslySetInnerHTML={{
                            __html: t('home.hero.description')
                                .replace('<strong>', '<span class="text-teal-400 font-semibold">')
                                .replace('</strong>', '</span>')
                        }}
                    />

                    {/* CTA Buttons */}
                    <div className="flex flex-wrap gap-4 justify-center">
                        <Link
                            to="/doctor"
                            className="group relative inline-flex items-center gap-2 bg-teal-500 hover:bg-teal-600 text-white px-8 py-4 rounded-lg font-semibold transition-all transform hover:scale-105 shadow-lg shadow-teal-500/50 hover:shadow-teal-500/70"
                        >
                            <Stethoscope className="w-5 h-5" />
                            <span>{t('home.hero.doctorButton')}</span>
                            <span className="absolute inset-0 rounded-lg bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity"></span>
                        </Link>
                        <Link
                            to="/student"
                            className="group relative inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-8 py-4 rounded-lg font-semibold transition-all transform hover:scale-105 border border-white/20 hover:border-teal-500/50"
                        >
                            <GraduationCap className="w-5 h-5" />
                            <span>{t('home.hero.studentButton')}</span>
                        </Link>
                    </div>

                    {/* Additional info */}
                    <p className="text-center text-gray-500 text-sm mt-8">
                        {t('home.hero.trustedBy')}
                    </p>
                </div>
            </div>
        </div>
    );
};
