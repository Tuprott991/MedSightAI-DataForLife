import { createBrowserRouter, RouterProvider, Navigate, Outlet } from 'react-router-dom';
import { Layout } from '../components/layout';
import { AuthProvider, ProtectedRoute, RoleGuard } from '../components/authentication';
import Login from './Login';
import { Home } from './Home';
import { Doctor } from './Doctor';
import { Student } from './Student';
import { DoctorDetail } from './DoctorDetail';
import { StudentDetail } from './StudentDetail';
import { PacsVnaSettings } from './PacsVnaSettings';
import { NotFound } from './NotFound';

const router = createBrowserRouter([
    {
        path: '/',
        element: (
            <AuthProvider>
                <Outlet />
            </AuthProvider>
        ),
        children: [
            {
                index: true,
                element: <Navigate to="/login" replace />,
            },
            {
                path: 'login',
                element: <Login />
            },
            {
                path: '/',
                element: (
                    <ProtectedRoute>
                        <Layout />
                    </ProtectedRoute>
                ),
                children: [
                    {
                        path: 'home',
                        element: (
                            <RoleGuard allowedRoles={['doctor', 'admin']}>
                                <Home />
                            </RoleGuard>
                        )
                    },
                    {
                        path: 'doctor',
                        element: (
                            <RoleGuard allowedRoles={['doctor', 'admin']}>
                                <Doctor />
                            </RoleGuard>
                        )
                    },
                    {
                        path: 'doctor/:id',
                        element: (
                            <RoleGuard allowedRoles={['doctor', 'admin']}>
                                <DoctorDetail />
                            </RoleGuard>
                        )
                    },
                    {
                        path: 'student',
                        element: <Student />
                    },
                    {
                        path: 'student/:id',
                        element: <StudentDetail />
                    },
                    {
                        path: 'pacs-settings',
                        element: (
                            <RoleGuard allowedRoles={['admin']}>
                                <PacsVnaSettings />
                            </RoleGuard>
                        )
                    },
                ],
            },
            {
                path: '*',
                element: <NotFound />
            }
        ]
    }
]);

export const AppRouter = () => {
    return <RouterProvider router={router} />;
};
