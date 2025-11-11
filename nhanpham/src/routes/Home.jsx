import { Link } from 'react-router-dom';
import { Stethoscope, GraduationCap, Activity, Brain, Heart, Shield } from 'lucide-react';

export const Home = () => {
    return (
        <div className="min-h-screen bg-[#1b1b1b] text-white">
            {/* Hero Section */}
            <div className="relative overflow-hidden">
                <div className="absolute inset-0 bg-linear-to-br from-teal-500/20 to-transparent"></div>
                <div className="container mx-auto px-6 py-20 relative z-10">
                    <div className="max-w-4xl mx-auto text-center">
                        <div className="inline-flex items-center gap-2 bg-teal-500/20 border border-teal-500/30 rounded-full px-6 py-2 mb-6">
                            <Activity className="w-4 h-4 text-teal-500" />
                            <span className="text-sm text-teal-400">AI-Powered Healthcare Platform</span>
                        </div>
                        <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-linear-to-r from-white via-teal-200 to-teal-500 bg-clip-text text-transparent">
                            MedSightAI
                        </h1>
                        <p className="text-xl md:text-2xl text-gray-300 mb-4">
                            Data For Life
                        </p>
                        <p className="text-lg text-gray-400 max-w-2xl mx-auto mb-12">
                            Revolutionizing healthcare with artificial intelligence.
                            Empowering doctors and students with cutting-edge diagnostic tools and medical insights.
                        </p>
                        <div className="flex flex-wrap gap-4 justify-center">
                            <Link
                                to="/doctor"
                                className="group relative inline-flex items-center gap-2 bg-teal-500 hover:bg-teal-600 text-white px-8 py-4 rounded-lg font-semibold transition-all transform hover:scale-105 shadow-lg shadow-teal-500/50"
                            >
                                <Stethoscope className="w-5 h-5" />
                                For Doctors
                                <span className="absolute inset-0 rounded-lg bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity"></span>
                            </Link>
                            <Link
                                to="/student"
                                className="group relative inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-8 py-4 rounded-lg font-semibold transition-all transform hover:scale-105 border border-white/20"
                            >
                                <GraduationCap className="w-5 h-5" />
                                For Students
                            </Link>
                        </div>
                    </div>
                </div>
            </div>

            {/* Features Section */}
            <div className="container mx-auto px-6 py-20">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-4xl font-bold mb-4">Why Choose MedSightAI?</h2>
                    <p className="text-gray-400 text-lg">Advanced features designed for modern healthcare</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {/* Feature 1 */}
                    <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 hover:border-teal-500/50 transition-all duration-300 hover:transform hover:scale-105">
                        <div className="w-14 h-14 bg-teal-500/20 rounded-lg flex items-center justify-center mb-6 group-hover:bg-teal-500/30 transition-colors">
                            <Brain className="w-7 h-7 text-teal-500" />
                        </div>
                        <h3 className="text-xl font-bold mb-3">AI-Powered Diagnosis</h3>
                        <p className="text-gray-400">
                            Leverage advanced machine learning algorithms for accurate medical diagnoses and treatment recommendations.
                        </p>
                    </div>

                    {/* Feature 2 */}
                    <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 hover:border-teal-500/50 transition-all duration-300 hover:transform hover:scale-105">
                        <div className="w-14 h-14 bg-teal-500/20 rounded-lg flex items-center justify-center mb-6 group-hover:bg-teal-500/30 transition-colors">
                            <Heart className="w-7 h-7 text-teal-500" />
                        </div>
                        <h3 className="text-xl font-bold mb-3">Patient Care Management</h3>
                        <p className="text-gray-400">
                            Comprehensive patient records, treatment history, and appointment scheduling in one unified platform.
                        </p>
                    </div>

                    {/* Feature 3 */}
                    <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 hover:border-teal-500/50 transition-all duration-300 hover:transform hover:scale-105">
                        <div className="w-14 h-14 bg-teal-500/20 rounded-lg flex items-center justify-center mb-6 group-hover:bg-teal-500/30 transition-colors">
                            <Shield className="w-7 h-7 text-teal-500" />
                        </div>
                        <h3 className="text-xl font-bold mb-3">Secure & Compliant</h3>
                        <p className="text-gray-400">
                            Enterprise-grade security with full compliance to healthcare data protection standards and regulations.
                        </p>
                    </div>

                    {/* Feature 4 */}
                    <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 hover:border-teal-500/50 transition-all duration-300 hover:transform hover:scale-105">
                        <div className="w-14 h-14 bg-teal-500/20 rounded-lg flex items-center justify-center mb-6 group-hover:bg-teal-500/30 transition-colors">
                            <GraduationCap className="w-7 h-7 text-teal-500" />
                        </div>
                        <h3 className="text-xl font-bold mb-3">Medical Education</h3>
                        <p className="text-gray-400">
                            Interactive learning modules, virtual simulations, and comprehensive medical resources for students.
                        </p>
                    </div>

                    {/* Feature 5 */}
                    <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 hover:border-teal-500/50 transition-all duration-300 hover:transform hover:scale-105">
                        <div className="w-14 h-14 bg-teal-500/20 rounded-lg flex items-center justify-center mb-6 group-hover:bg-teal-500/30 transition-colors">
                            <Activity className="w-7 h-7 text-teal-500" />
                        </div>
                        <h3 className="text-xl font-bold mb-3">Real-time Analytics</h3>
                        <p className="text-gray-400">
                            Monitor patient vitals, track treatment progress, and analyze health trends with real-time data visualization.
                        </p>
                    </div>

                    {/* Feature 6 */}
                    <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 hover:border-teal-500/50 transition-all duration-300 hover:transform hover:scale-105">
                        <div className="w-14 h-14 bg-teal-500/20 rounded-lg flex items-center justify-center mb-6 group-hover:bg-teal-500/30 transition-colors">
                            <Stethoscope className="w-7 h-7 text-teal-500" />
                        </div>
                        <h3 className="text-xl font-bold mb-3">Expert Collaboration</h3>
                        <p className="text-gray-400">
                            Connect with medical professionals worldwide, share insights, and collaborate on complex cases.
                        </p>
                    </div>
                </div>
            </div>

            {/* CTA Section */}
            <div className="container mx-auto px-6 py-20">
                <div className="bg-linear-to-r from-teal-500/20 to-teal-600/20 border border-teal-500/30 rounded-2xl p-12 text-center">
                    <h2 className="text-3xl md:text-4xl font-bold mb-4">Ready to Transform Healthcare?</h2>
                    <p className="text-gray-300 text-lg mb-8 max-w-2xl mx-auto">
                        Join thousands of healthcare professionals already using MedSightAI to improve patient outcomes and advance medical knowledge.
                    </p>
                    <div className="flex flex-wrap gap-4 justify-center">
                        <Link
                            to="/doctor"
                            className="inline-flex items-center gap-2 bg-teal-500 hover:bg-teal-600 text-white px-8 py-4 rounded-lg font-semibold transition-all transform hover:scale-105"
                        >
                            Get Started as Doctor
                        </Link>
                        <Link
                            to="/student"
                            className="inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-8 py-4 rounded-lg font-semibold transition-all border border-white/20"
                        >
                            Start Learning
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Home;
