import { useNavigate } from 'react-router-dom';
import { useAuth } from '../components/authentication';
import { LoginForm } from '../components/authentication';
import { useEffect } from 'react';

const Login = () => {
    const { login, isAuthenticated, user } = useAuth();
    const navigate = useNavigate();

    // Redirect if already authenticated
    useEffect(() => {
        if (isAuthenticated) {
            if (user?.role === 'student') {
                navigate('/student');
            } else {
                navigate('/home');
            }
        }
    }, [isAuthenticated, user, navigate]);

    const handleLogin = (userData) => {
        login(userData);
        // Redirect based on role
        if (userData.role === 'student') {
            navigate('/student');
        } else {
            navigate('/home');
        }
    };

    return (
        <div className="min-h-screen bg-linear-to-br from-gray-900 via-[#0a0a0a] to-teal-900/20 flex items-center justify-center px-4 relative overflow-hidden">
            {/* Animated Background */}
            <div className="absolute inset-0 overflow-hidden">
                {/* Gradient Orbs */}
                <div className="absolute top-0 -left-4 w-96 h-96 bg-teal-500 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob"></div>
                <div className="absolute top-0 -right-4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-2000"></div>
                <div className="absolute -bottom-8 left-20 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-4000"></div>

                {/* Medical Cross Pattern */}
                <div className="absolute inset-0 opacity-10">
                    <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <pattern id="medical-pattern" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
                                <path d="M45 30 L45 45 L30 45 L30 55 L45 55 L45 70 L55 70 L55 55 L70 55 L70 45 L55 45 L55 30 Z" fill="currentColor" className="text-teal-500" />
                            </pattern>
                        </defs>
                        <rect width="100%" height="100%" fill="url(#medical-pattern)" />
                    </svg>
                </div>

                {/* Grid Pattern */}
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff12_1px,transparent_1px),linear-gradient(to_bottom,#ffffff12_1px,transparent_1px)] bg-size-[4rem_4rem]"></div>
            </div>

            <div className="w-full max-w-md relative z-10">
                {/* Logo/Brand Section */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-20 h-20 bg-linear-to-br from-teal-500 to-teal-600 rounded-2xl mb-4 shadow-lg shadow-teal-500/20">
                        <svg
                            className="w-12 h-12 text-white"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                            />
                        </svg>
                    </div>
                    <h1 className="text-3xl font-bold text-white mb-2">MedSightAI</h1>
                    <p className="text-gray-400">Hệ thống hỗ trợ chẩn đoán y tế thông minh</p>
                </div>

                {/* Login Card */}
                <div className="bg-[#141414]/90 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/10 p-8">
                    <h2 className="text-2xl font-bold text-white mb-6">Đăng nhập</h2>
                    <LoginForm onLogin={handleLogin} />
                </div>

                {/* Footer */}
                <div className="text-center mt-8">
                    <p className="text-sm text-gray-500">
                        © 2025 MedSightAI. Bảo mật và riêng tư được đảm bảo.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Login;
