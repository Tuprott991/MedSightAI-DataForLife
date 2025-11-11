import { Outlet, Link, useLocation } from "react-router-dom";
import { FloatingDirection } from "./FloatingDirection";
import { Stethoscope, GraduationCap, Home as HomeIcon } from "lucide-react";

export const Layout = () => {
    const location = useLocation();

    const isActive = (path) => {
        return location.pathname === path || (location.pathname === '/' && path === '/home');
    };

    return (
        <div className="min-h-screen flex flex-col bg-[#1b1b1b]">
            {/* Navigation Bar */}
            <nav className="bg-[#1b1b1b] border-b border-white/10 backdrop-blur-lg sticky top-0 z-50">
                <div className="container mx-auto px-6">
                    <div className="flex items-center justify-between h-16">
                        <div className="flex items-center gap-2">
                            <div className="w-8 h-8 bg-teal-500 rounded-lg flex items-center justify-center">
                                <span className="text-white font-bold text-sm">M</span>
                            </div>
                            <h1 className="text-xl font-bold text-white">MedSightAI</h1>
                        </div>
                        <div className="flex gap-2">
                            <Link
                                to="/home"
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${isActive('/home')
                                        ? 'bg-teal-500 text-white shadow-lg shadow-teal-500/50'
                                        : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                    }`}
                            >
                                <HomeIcon className="w-4 h-4" />
                                <span className="font-medium">Home</span>
                            </Link>
                            <Link
                                to="/doctor"
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${isActive('/doctor')
                                        ? 'bg-teal-500 text-white shadow-lg shadow-teal-500/50'
                                        : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                    }`}
                            >
                                <Stethoscope className="w-4 h-4" />
                                <span className="font-medium">Doctor</span>
                            </Link>
                            <Link
                                to="/student"
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${isActive('/student')
                                        ? 'bg-teal-500 text-white shadow-lg shadow-teal-500/50'
                                        : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                    }`}
                            >
                                <GraduationCap className="w-4 h-4" />
                                <span className="font-medium">Student</span>
                            </Link>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="flex-1">
                <Outlet />
            </main>

            <FloatingDirection />
        </div>
    )
}