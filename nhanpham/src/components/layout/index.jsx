import { Outlet, Link, useLocation, useParams } from "react-router-dom";
import { FloatingDirection } from "./FloatingDirection";
import { Stethoscope, GraduationCap, Home as HomeIcon, ArrowLeft, Settings, HelpCircle, Bell, User, PanelLeft, PanelLeftClose } from "lucide-react";
import { patientsData } from "../../constants/patients";
import { createContext, useContext, useState } from "react";

// Create context for sidebar collapse state
const SidebarContext = createContext();
export const useSidebar = () => useContext(SidebarContext);

export const Layout = () => {
    const location = useLocation();
    const params = useParams();
    const [isLeftCollapsed, setIsLeftCollapsed] = useState(false);

    const isActive = (path) => {
        return location.pathname === path || (location.pathname === '/' && path === '/home');
    };

    // Check if we're on a detail page
    const isDetailPage = location.pathname.startsWith('/doctor/') || location.pathname.startsWith('/student/');
    const isDoctorDetail = location.pathname.startsWith('/doctor/') && params.id;
    const isStudentDetail = location.pathname.startsWith('/student/') && params.id;

    // Get patient info if on detail page
    let patient = null;
    let backPath = '/';
    let backLabel = 'Back';

    if (isDoctorDetail) {
        patient = patientsData.find(p => p.id === parseInt(params.id));
        backPath = '/doctor';
        backLabel = 'Back to Patient Records';
    } else if (isStudentDetail) {
        patient = patientsData.find(p => p.id === parseInt(params.id));
        backPath = '/student';
        backLabel = 'Back to Student Records';
    }

    return (
        <SidebarContext.Provider value={{ isLeftCollapsed, setIsLeftCollapsed }}>
            <div className="min-h-screen flex flex-col bg-[#1b1b1b]">
                {/* Navigation Bar */}
                <nav className="bg-[#1b1b1b] border-b border-white/10 backdrop-blur-lg sticky top-0 z-50">
                    <div className="container mx-auto px-6">
                        <div className="flex items-center justify-between h-16">
                            {/* Left Side - Logo or Back Button */}
                            {isDetailPage ? (
                                <Link
                                    to={backPath}
                                    className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
                                >
                                    <ArrowLeft className="w-4 h-4" />
                                    <span className="text-sm">{backLabel}</span>
                                </Link>
                            ) : (
                                <div className="flex items-center gap-2">
                                    <div className="w-8 h-8 bg-teal-500 rounded-lg flex items-center justify-center">
                                        <span className="text-white font-bold text-sm">M</span>
                                    </div>
                                    <h1 className="text-xl font-bold text-white">MedSightAI</h1>
                                </div>
                            )}

                            {/* Center - Patient Info (only on detail pages) */}
                            {isDetailPage && patient && (
                                <div className="absolute left-1/2 transform -translate-x-1/2 text-center">
                                    <h2 className="text-lg font-semibold text-white">{patient.name}</h2>
                                    <p className="text-xs text-gray-400">{patient.diagnosis}</p>
                                </div>
                            )}

                            {/* Right Side - Navigation Links or Action Icons */}
                            {!isDetailPage ? (
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
                            ) : (
                                <div className="flex items-center gap-3">
                                    {/* Settings */}
                                    <button className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors">
                                        <Settings className="w-5 h-5" />
                                    </button>

                                    {/* Help */}
                                    <button className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors">
                                        <HelpCircle className="w-5 h-5" />
                                    </button>

                                    {/* Notifications */}
                                    <button className="relative p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors">
                                        <Bell className="w-5 h-5" />
                                        {/* Notification Badge */}
                                        <span className="absolute top-1 right-1 w-4 h-4 bg-red-500 text-white text-xs flex items-center justify-center rounded-full border-2 border-[#1b1b1b]">
                                            2
                                        </span>
                                    </button>

                                    {/* Avatar */}
                                    <button className="p-1 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors">
                                        <div className="w-8 h-8 bg-teal-500 rounded-full flex items-center justify-center">
                                            <User className="w-5 h-5 text-white" />
                                        </div>
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </nav>

                {/* Main Content */}
                <main className="flex-1">
                    <Outlet />
                </main>

                {/* FloatingDirection - Hidden on detail pages */}
                {!isDetailPage && <FloatingDirection />}
            </div>
        </SidebarContext.Provider>
    )
}