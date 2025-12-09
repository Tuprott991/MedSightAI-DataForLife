import { Navigate } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { useTranslation } from 'react-i18next';

export const RoleGuard = ({ children, allowedRoles }) => {
    const { user, isLoading } = useAuth();
    const { t } = useTranslation();

    if (isLoading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-[#0a0a0a]">
                <div className="text-white text-lg">{t('common.loading')}</div>
            </div>
        );
    }

    if (!user) {
        return <Navigate to="/login" replace />;
    }

    if (allowedRoles && !allowedRoles.includes(user.role)) {
        return <Navigate to="/home" replace />;
    }

    return children;
};
