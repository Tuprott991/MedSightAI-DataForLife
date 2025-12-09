import { useTranslation } from 'react-i18next';
import { Languages } from 'lucide-react';
import { useState } from 'react';

export const LanguageSwitcher = () => {
    const { i18n } = useTranslation();
    const [isOpen, setIsOpen] = useState(false);

    const languages = [
        { code: 'vi', name: 'Tiếng Việt', flag: '🇻🇳' },
        { code: 'en', name: 'English', flag: '🇺🇸' }
    ];

    const currentLanguage = languages.find(lang => lang.code === i18n.language) || languages[0];

    const changeLanguage = (langCode) => {
        i18n.changeLanguage(langCode);
        setIsOpen(false);
    };

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-teal-500/10 hover:bg-teal-500/20 transition-all border border-teal-500/30 hover:border-teal-500/50 text-teal-400 hover:text-teal-300"
                title="Change Language"
            >
                <Languages className="w-4 h-4" />
                <span className="text-sm font-medium">{currentLanguage.flag}</span>
            </button>

            {isOpen && (
                <>
                    <div
                        className="fixed inset-0 z-40"
                        onClick={() => setIsOpen(false)}
                    />
                    <div className="absolute right-0 mt-2 w-48 bg-[#1a1a1a] border border-teal-500/30 rounded-lg shadow-2xl shadow-teal-500/10 z-50 overflow-hidden">
                        {languages.map((lang) => (
                            <button
                                key={lang.code}
                                onClick={() => changeLanguage(lang.code)}
                                className={`w-full flex items-center gap-3 px-4 py-3 transition-all ${currentLanguage.code === lang.code
                                        ? 'bg-teal-500/20 text-teal-400 border-l-2 border-teal-500'
                                        : 'text-gray-300 hover:bg-teal-500/10 hover:text-teal-400'
                                    }`}
                            >
                                <span className="text-xl">{lang.flag}</span>
                                <span className="text-sm font-medium">{lang.name}</span>
                                {currentLanguage.code === lang.code && (
                                    <span className="ml-auto text-teal-400 font-bold">✓</span>
                                )}
                            </button>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
};
