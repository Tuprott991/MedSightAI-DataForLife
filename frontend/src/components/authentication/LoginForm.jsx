import { useState } from 'react';
import { User, Lock, Eye, EyeOff } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const LoginForm = ({ onLogin }) => {
    const { t } = useTranslation();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        // Validation
        if (!username.trim()) {
            setError(t('auth.login.emailPlaceholder'));
            setIsLoading(false);
            return;
        }

        if (!password) {
            setError(t('auth.login.passwordPlaceholder'));
            setIsLoading(false);
            return;
        }

        // Mock authentication with specific accounts
        setTimeout(() => {
            const validAccounts = {
                'admin@example.com': {
                    username: 'admin@example.com',
                    role: 'admin',
                    name: t('auth.login.admin')
                },
                'doctor@example.com': {
                    username: 'doctor@example.com',
                    role: 'doctor',
                    name: t('auth.login.doctor')
                },
                'student@example.com': {
                    username: 'student@example.com',
                    role: 'student',
                    name: t('auth.login.student')
                }
            };

            const account = validAccounts[username.toLowerCase()];

            if (account && password === '123456') {
                onLogin(account);
            } else {
                setError('Email hoặc mật khẩu không đúng');
            }
            setIsLoading(false);
        }, 1000);
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email Field */}
            <div>
                <label htmlFor="username" className="block text-sm font-medium text-gray-300 mb-2">
                    {t('auth.login.email')}
                </label>
                <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <User className="h-5 w-5 text-gray-400" />
                    </div>
                    <input
                        id="username"
                        type="email"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        className="block w-full pl-10 pr-3 py-3 border border-white/10 rounded-lg bg-[#1a1a1a] text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all"
                        placeholder={t('auth.login.emailPlaceholder')}
                        disabled={isLoading}
                    />
                </div>
            </div>

            {/* Password Field */}
            <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                    {t('auth.login.password')}
                </label>
                <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Lock className="h-5 w-5 text-gray-400" />
                    </div>
                    <input
                        id="password"
                        type={showPassword ? 'text' : 'password'}
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="block w-full pl-10 pr-12 py-3 border border-white/10 rounded-lg bg-[#1a1a1a] text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all"
                        placeholder={t('auth.login.passwordPlaceholder')}
                        disabled={isLoading}
                    />
                    <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white transition-colors"
                        disabled={isLoading}
                    >
                        {showPassword ? (
                            <EyeOff className="h-5 w-5" />
                        ) : (
                            <Eye className="h-5 w-5" />
                        )}
                    </button>
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                    <p className="text-sm text-red-400">{error}</p>
                </div>
            )}

            {/* Submit Button */}
            <button
                type="submit"
                disabled={isLoading}
                className="w-full py-3 px-4 bg-teal-500 hover:bg-teal-600 disabled:bg-teal-500/50 text-white font-medium rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2 focus:ring-offset-[#0a0a0a] disabled:cursor-not-allowed"
            >
                {isLoading ? t('common.loading') : t('auth.login.signIn')}
            </button>

            {/* Additional Links */}
            <div className="text-center">
                <a href="#" className="text-sm text-teal-400 hover:text-teal-300 transition-colors">
                    {t('auth.login.forgotPassword')}
                </a>
            </div>
        </form>
    );
};
