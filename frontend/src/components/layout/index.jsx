import { Outlet, Link, useLocation, useParams, useNavigate } from "react-router-dom";
import { FloatingDirection } from "./FloatingDirection";
import { Stethoscope, GraduationCap, Home as HomeIcon, ArrowLeft, Settings, HelpCircle, Bell, User, PanelLeft, PanelLeftClose, LogOut, Database } from "lucide-react";
import { patientsData } from "../../constants/patients";
import { createContext, useContext, useState } from "react";
import { useAuth } from "../authentication";

// Create context for sidebar collapse state
const SidebarContext = createContext();
export const useSidebar = () => useContext(SidebarContext);

export const Layout = () => {
    const location = useLocation();
    const params = useParams();
    const navigate = useNavigate();
    const [isLeftCollapsed, setIsLeftCollapsed] = useState(false);
    const [showLogoutModal, setShowLogoutModal] = useState(false);
    const { user, logout } = useAuth();

    const handleLogout = () => {
        setShowLogoutModal(true);
    };

    const confirmLogout = () => {
        setShowLogoutModal(false);
        logout();
        navigate('/login');
    };

    const cancelLogout = () => {
        setShowLogoutModal(false);
    };

    const isActive = (path) => {
        return location.pathname === path || (location.pathname === '/' && path === '/home');
    };

    // Check if we're on a detail page
    const isDetailPage = location.pathname.startsWith('/doctor/') || location.pathname.startsWith('/student/');
    const isPacsSettingsPage = location.pathname === '/pacs-settings';
    const isDoctorDetail = location.pathname.startsWith('/doctor/') && params.id;
    const isStudentDetail = location.pathname.startsWith('/student/') && params.id;

    // Get patient info if on detail page
    let patient = null;
    let backPath = '/';
    let backLabel = 'Back';

    if (isDoctorDetail) {
        patient = patientsData.find(p => p.id === parseInt(params.id));
        backPath = '/doctor';
        backLabel = 'Quay Lại Danh Sách Bệnh Nhân';
    } else if (isStudentDetail) {
        patient = patientsData.find(p => p.id === parseInt(params.id));
        backPath = '/student';
        backLabel = 'Quay Lại Danh Sách Bệnh Nhân';
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
                                <Link to="/home" className="flex items-center gap-2 hover:opacity-80 transition-opacity cursor-pointer">
                                    <div className="w-8 h-8 bg-teal-500 rounded-lg flex items-center justify-center">
                                        <span className="text-white font-bold text-sm">M</span>
                                    </div>
                                    <h1 className="text-xl font-bold text-white">MedSightAI</h1>
                                </Link>
                            )}

                            {/* Center - Patient Info (only on doctor detail pages, hidden for students) */}
                            {isDetailPage && patient && isDoctorDetail && !isStudentDetail && (
                                <div className="absolute left-1/2 transform -translate-x-1/2 text-center">
                                    <h2 className="text-lg font-semibold text-white">{patient.name}</h2>
                                    {patient.status !== 'Admitted' && (
                                        <p className="text-xs text-gray-400">{patient.diagnosis}</p>
                                    )}
                                </div>
                            )}

                            {/* Right Side - Navigation Links or Action Icons */}
                            {!isDetailPage ? (
                                <div className="flex gap-2">
                                    {(user?.role === 'doctor' || user?.role === 'admin') && (
                                        <>
                                            <Link
                                                to="/home"
                                                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${isActive('/home')
                                                    ? 'bg-teal-500 text-white shadow-lg shadow-teal-500/50'
                                                    : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                                    }`}
                                            >
                                                <HomeIcon className="w-4 h-4" />
                                                <span className="font-medium">Trang Chủ</span>
                                            </Link>
                                            <Link
                                                to="/doctor"
                                                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${isActive('/doctor')
                                                    ? 'bg-teal-500 text-white shadow-lg shadow-teal-500/50'
                                                    : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                                    }`}
                                            >
                                                <Stethoscope className="w-4 h-4" />
                                                <span className="font-medium">Bác Sĩ</span>
                                            </Link>
                                        </>
                                    )}
                                    <Link
                                        to="/student"
                                        className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${isActive('/student')
                                            ? 'bg-teal-500 text-white shadow-lg shadow-teal-500/50'
                                            : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                            }`}
                                    >
                                        <GraduationCap className="w-4 h-4" />
                                        <span className="font-medium">Sinh Viên</span>
                                    </Link>
                                    {user?.role === 'admin' && (
                                        <Link
                                            to="/pacs-settings"
                                            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${isActive('/pacs-settings')
                                                ? 'bg-teal-500 text-white shadow-lg shadow-teal-500/50'
                                                : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                                }`}
                                        >
                                            <Database className="w-4 h-4" />
                                            <span className="font-medium">Cài đặt PACs/VNA</span>
                                        </Link>
                                    )}
                                    <button
                                        onClick={handleLogout}
                                        className="flex items-center gap-2 px-4 py-2 rounded-lg text-gray-300 hover:bg-red-500/10 hover:text-red-400 transition-all"
                                        title="Đăng xuất"
                                    >
                                        <LogOut className="w-4 h-4" />
                                        <span className="font-medium">Đăng xuất</span>
                                    </button>
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

                                    {/* User Menu */}
                                    <div className="flex items-center gap-2 ml-2">
                                        <div className="text-right mr-2">
                                            <p className="text-sm font-medium text-white">{user?.name}</p>
                                            <p className="text-xs text-gray-400">
                                                {user?.role === 'doctor' ? 'Bác sĩ' : user?.role === 'admin' ? 'Quản trị viên' : 'Sinh viên'}
                                            </p>
                                        </div>
                                        <div className="w-8 h-8 bg-teal-500 rounded-full flex items-center justify-center">
                                            <User className="w-5 h-5 text-white" />
                                        </div>
                                        <button
                                            onClick={handleLogout}
                                            className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                                            title="Đăng xuất"
                                        >
                                            <LogOut className="w-5 h-5" />
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </nav>

                {/* Main Content */}
                <main className="flex-1">
                    <Outlet />
                </main>

                {/* FloatingDirection - Hidden on detail pages and settings page */}
                {!isDetailPage && !isPacsSettingsPage && <FloatingDirection />}

                {/* Logout Confirmation Modal */}
                {showLogoutModal && (
                    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="w-12 h-12 bg-red-500/10 rounded-full flex items-center justify-center">
                                    <LogOut className="w-6 h-6 text-red-500" />
                                </div>
                                <div>
                                    <h3 className="text-lg font-semibold text-white">Xác nhận đăng xuất</h3>
                                    <p className="text-sm text-gray-400">Bạn có chắc chắn muốn đăng xuất?</p>
                                </div>
                            </div>
                            <div className="flex gap-3 mt-6">
                                <button
                                    onClick={cancelLogout}
                                    className="flex-1 px-4 py-2.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors font-medium"
                                >
                                    Hủy
                                </button>
                                <button
                                    onClick={confirmLogout}
                                    className="flex-1 px-4 py-2.5 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors font-medium shadow-lg shadow-red-500/30"
                                >
                                    Đăng xuất
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </SidebarContext.Provider>
    )
}