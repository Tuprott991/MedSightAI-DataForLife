import { User, Droplet, Image as ImageIcon } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { getTranslatedDiagnosis, getTranslatedGender } from '../../utils/diagnosisHelper';
import { DicomImage } from './DicomImage';
import { getProxiedImageUrl } from '../../services/patientApi';

export const PatientCard = ({ patient, isStudentView = false }) => {
    const { t, i18n } = useTranslation();
    const location = useLocation();
    const isStudentPage = location.pathname.includes('/student') || isStudentView;

    // Get image URL - either from patient.image or latest_case
    const rawImageUrl = patient.image || patient.latest_case?.image_path || patient.latest_case?.processed_img_path;
    const processingStatus = patient.latest_case
        ? (patient.latest_case.processed_img_path ? 'processed' : 'unprocessed')
        : null;
    
    // Use the S3 URL exactly as it is stored in the database.
    const imageUrl = getProxiedImageUrl(rawImageUrl);

    const processingStatusLabel = processingStatus
        ? t(`doctor.filters.${processingStatus}`)
        : null;
    const processingStatusColor = processingStatus === 'processed'
        ? 'bg-green-500/25 text-green-300 border-green-500/40'
        : 'bg-red-500/25 text-red-300 border-red-500/40';

    return (
        <div className="group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl overflow-hidden transition-all duration-300 hover:border-teal-500/30 hover:shadow-lg hover:shadow-teal-500/10">
            {/* Patient Image */}
            <div className="relative h-48 overflow-hidden bg-gradient-to-br from-teal-500/20 to-emerald-500/20">
                {imageUrl ? (
                    <DicomImage
                        src={imageUrl}
                        alt={patient.name}
                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                    />
                ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center text-center p-4">
                        <ImageIcon className="w-16 h-16 text-gray-600 mb-2" />
                        <p className="text-gray-500 text-sm">Chưa có ảnh</p>
                        <p className="text-gray-600 text-xs mt-1">ID: {patient.id.substring(0, 8)}</p>
                    </div>
                )}
                {processingStatusLabel && (
                    <div className="absolute top-3 right-3">
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${processingStatusColor}`}>
                            {processingStatusLabel}
                        </span>
                    </div>
                )}
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
                        {patient.diagnosis && (
                            <p className="text-sm text-teal-400 mb-4">{getTranslatedDiagnosis(patient.diagnosis, t)}</p>
                        )}

                        <div className="space-y-2">
                            {patient.age && patient.gender && (
                                <div className="flex items-center gap-2 text-sm text-gray-400">
                                    <User className="w-4 h-4" />
                                    <span>{patient.age} {t('doctor.patientCard.years')} • {getTranslatedGender(patient.gender, t)}</span>
                                </div>
                            )}

                            {patient.blood_type && (
                                <div className="flex items-center gap-2 text-sm text-gray-400">
                                    <Droplet className="w-4 h-4" />
                                    <span>{i18n.language === 'vi' ? 'Nhóm máu' : 'Blood Type'}: {patient.blood_type}</span>
                                </div>
                            )}
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
