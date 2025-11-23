import { Calendar, User, Activity, Droplet } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

export const PatientCard = ({ patient }) => {
    const location = useLocation();
    const isStudentPage = location.pathname.includes('/student');

    const getStatusColor = (status) => {
        switch (status) {
            case 'Critical':
                return 'bg-red-500/20 text-red-400 border-red-500/30';
            case 'Under Treatment':
                return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
            case 'Stable':
                return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
            default:
                return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
        }
    };

    return (
        <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl overflow-hidden hover:bg-white/10 hover:border-teal-500/50 transition-all duration-300 hover:transform hover:scale-105 cursor-pointer">
            {/* Patient Image */}
            <div className="relative h-48 overflow-hidden bg-linear-to-br from-teal-500/20 to-emerald-500/20">
                <img
                    src={patient.image}
                    alt={patient.name}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                />
                <div className="absolute top-3 right-3">
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getStatusColor(patient.status)}`}>
                        {patient.status}
                    </span>
                </div>
            </div>

            {/* Patient Info */}
            <div className="p-5">
                <h3 className="text-lg font-bold text-white mb-1">{patient.name}</h3>
                <p className="text-sm text-teal-400 mb-4">{patient.diagnosis}</p>

                <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm text-gray-400">
                        <User className="w-4 h-4" />
                        <span>{patient.age} years • {patient.gender}</span>
                    </div>

                    <div className="flex items-center gap-2 text-sm text-gray-400">
                        <Droplet className="w-4 h-4" />
                        <span>Blood Type: {patient.bloodType}</span>
                    </div>

                    <div className="flex items-center gap-2 text-sm text-gray-400">
                        <Calendar className="w-4 h-4" />
                        <span>Admitted: {new Date(patient.admissionDate).toLocaleDateString()}</span>
                    </div>

                    <div className="flex items-center gap-2 text-sm text-gray-400">
                        <Activity className="w-4 h-4" />
                        <span>Last Visit: {new Date(patient.lastVisit).toLocaleDateString()}</span>
                    </div>
                </div>

                <Link
                    to={isStudentPage ? `/student/${patient.id}` : `/doctor/${patient.id}`}
                    className="mt-4 w-full bg-teal-500/20 hover:bg-teal-500 text-teal-400 hover:text-white border border-teal-500/30 hover:border-teal-500 px-4 py-2 rounded-lg transition-all font-medium block text-center"
                >
                    View Details
                </Link>
            </div>
        </div>
    );
};
