import { Calendar, User, Activity, Droplet } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

export const PatientCard = ({ patient }) => {
    const { t, i18n } = useTranslation();
    const location = useLocation();
    const isStudentPage = location.pathname.includes('/student');

    const getStatusColor = (status) => {
        switch (status) {
            case 'Critical':
                return 'bg-red-600/30 text-white border-red-600/40'; // Nguy kịch - Đỏ
            case 'Under Treatment':
                return 'bg-blue-500/30 text-white border-blue-500/40'; // Đang điều trị - Xanh dương
            case 'Stable':
                return 'bg-green-500/30 text-white border-green-500/40'; // Ổn định - Xanh lá
            case 'Admitted':
                return 'bg-teal-500/30 text-white border-teal-500/40'; // Tiếp nhận - Màu logo (teal)
            default:
                return 'bg-gray-500/30 text-white border-gray-500/40';
        }
    };

    const getStatusText = (status) => {
        switch (status) {
            case 'Critical':
                return t('doctorDetail.patientInfo.critical');
            case 'Under Treatment':
                return t('doctorDetail.patientInfo.stable');
            case 'Stable':
                return t('doctorDetail.patientInfo.stable');
            case 'Admitted':
                return t('doctorDetail.patientInfo.improving');
            default:
                return status;
        }
    };

    return (
        <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl overflow-hidden transition-all duration-300">
            {/* Patient Image */}
            <div className="relative h-48 overflow-hidden bg-linear-to-br from-teal-500/20 to-emerald-500/20">
                <img
                    src={patient.image}
                    alt={patient.name}
                    className="w-full h-full object-cover transition-transform duration-300"
                />
                <div className="absolute top-3 right-3">
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getStatusColor(patient.status)}`}>
                        {getStatusText(patient.status)}
                    </span>
                </div>
            </div>

            {/* Patient Info */}
            <div className="p-5">
                {isStudentPage ? (
                    // Student view - chỉ hiện button
                    <Link
                        to={`/student/${patient.id}`}
                        className="w-full bg-teal-500/20 hover:bg-teal-500 text-teal-400 hover:text-white border border-teal-500/30 hover:border-teal-500 px-4 py-2 rounded-lg transition-all font-medium block text-center"
                    >
                        {t('doctor.patientCard.view')}
                    </Link>
                ) : (
                    // Doctor view - hiện đầy đủ thông tin
                    <>
                        <h3 className="text-lg font-bold text-white mb-1">{patient.name}</h3>
                        {patient.status !== 'Admitted' && (
                            <p className="text-sm text-teal-400 mb-4">{patient.diagnosis}</p>
                        )}

                        <div className="space-y-2">
                            <div className="flex items-center gap-2 text-sm text-gray-400">
                                <User className="w-4 h-4" />
                                <span>{patient.age} {t('doctor.patientCard.years')} • {patient.gender === 'Male' ? t('doctorDetail.patientInfo.male') : t('doctorDetail.patientInfo.female')}</span>
                            </div>

                            <div className="flex items-center gap-2 text-sm text-gray-400">
                                <Droplet className="w-4 h-4" />
                                <span>{new Date(patient.admissionDate).toLocaleDateString(i18n.language === 'vi' ? 'vi-VN' : 'en-US')}</span>
                            </div>

                            <div className="flex items-center gap-2 text-sm text-gray-400">
                                <Calendar className="w-4 h-4" />
                                <span>{new Date(patient.admissionDate).toLocaleDateString(i18n.language === 'vi' ? 'vi-VN' : 'en-US')}</span>
                            </div>

                            <div className="flex items-center gap-2 text-sm text-gray-400">
                                <Activity className="w-4 h-4" />
                                <span>{new Date(patient.lastVisit).toLocaleDateString(i18n.language === 'vi' ? 'vi-VN' : 'en-US')}</span>
                            </div>
                        </div>

                        <Link
                            to={`/doctor/${patient.id}`}
                            className="mt-4 w-full bg-teal-500/20 hover:bg-teal-500 text-teal-400 hover:text-white border border-teal-500/30 hover:border-teal-500 px-4 py-2 rounded-lg transition-all font-medium block text-center"
                        >
                            {t('doctor.patientCard.view')}
                        </Link>
                    </>
                )}
            </div>
        </div>
    );
};
