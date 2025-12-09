import { Navigate } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { useTranslation } from 'react-i18next';

export const ProtectedRoute = ({ children }) => {
    const { isAuthenticated, isLoading } = useAuth();
    const { t } = useTranslation();

    if (isLoading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-[#0a0a0a]">
                <div className="text-white text-lg">{t('common.loading')}</div>
            </div>
        );
    }

    if (!isAuthenticated) {
        return <Navigate to="/login" replace />;
    }

    return children;
};
