import { createRoot } from 'react-dom/client';
import { AppRouter } from './routes';
import './index.css';
import './i18n/config';

const container = document.getElementById("root");
const root = createRoot(container);
root.render(<AppRouter />);
