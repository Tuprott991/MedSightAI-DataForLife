import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { Layout } from '../components/layout';
import { Home } from './Home';
import { Doctor } from './Doctor';
import { Student } from './Student';

const router = createBrowserRouter([
    {
        path: '/',
        element: <Layout />,
        children: [
            {
                path: '/',
                element: <Home />
            },
            {
                path: '/home',
                element: <Home />
            },
            {
                path: '/doctor',
                element: <Doctor />
            },
            {
                path: '/student',
                element: <Student />
            }
        ]
    }
]);

export const AppRouter = () => {
    return <RouterProvider router={router} />;
};
