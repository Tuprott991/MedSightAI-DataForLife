import { FileText, Download, Printer } from 'lucide-react';
import { getFindingImagePath } from '../../constants/medicalData';
import { useTranslation } from 'react-i18next';

export const ReportPDF = ({ reportData, patient, selectedImage, analysisData }) => {
    const { t } = useTranslation();
    if (!reportData || !patient) return null;

    // Get all xAI images for all findings
    const allXaiImages = [];
    if (analysisData?.findings) {
        analysisData.findings.forEach(finding => {
            const imagePath = getFindingImagePath(finding.text, patient.image);
            if (imagePath) {
                allXaiImages.push({
                    url: imagePath,
                    finding: finding.text,
                    severity: finding.severity
                });
            }
        });
    }

    const originalImage = patient.image;

    const handlePrint = () => {
        window.print();
    };

    const handleDownload = () => {
        // Create a temporary container for the report content
        const printContent = document.getElementById('report-content');
        if (!printContent) return;

        // Clone the content
        const clonedContent = printContent.cloneNode(true);

        // Create a new window for printing to PDF
        const printWindow = window.open('', '_blank');
        if (!printWindow) {
            alert(t('report.allowPopup'));
            return;
        }

        // Write the HTML structure
        printWindow.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>${t('report.reportTitle')} - ${patient.name}</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: white;
                        padding: 20px;
                    }
                    @page { size: A4; margin: 15mm; }
                    @media print {
                        body { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
                    }
                </style>
                <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            </head>
            <body>
                ${clonedContent.innerHTML}
            </body>
            </html>
        `);

        printWindow.document.close();

        // Wait for content to load, then trigger print dialog
        setTimeout(() => {
            printWindow.focus();
            printWindow.print();
            // Close window after printing or canceling
            setTimeout(() => printWindow.close(), 100);
        }, 500);
    };

    return (
        <div className="bg-white text-black">
            {/* Action Buttons - Hidden when printing */}
            <div className="flex gap-2 mb-4 print:hidden">
                <button
                    onClick={handlePrint}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                >
                    <Printer className="w-4 h-4" />
                    <span>{t('report.printReport')}</span>
                </button>
                <button
                    onClick={handleDownload}
                    className="flex items-center gap-2 px-4 py-2 bg-teal-500 text-white rounded-lg hover:bg-teal-600 transition-colors"
                >
                    <Download className="w-4 h-4" />
                    <span>{t('report.download')}</span>
                </button>
            </div>

            {/* PDF Content */}
            <div id="report-content" className="bg-white p-8 max-w-[210mm] mx-auto shadow-lg">
                {/* Header - Hospital Info */}
                <div className="relative mb-6 pb-4">
                    <div className="border-2 border-gray-400 rounded-lg p-8 relative">
                        <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-white px-3">
                            <span className="text-xs font-semibold text-gray-600 uppercase">{t('report.hospitalInfo')}</span>
                        </div>
                        <div className="text-center mt-2">
                            <p className="text-sm text-gray-500 italic">{t('report.hospitalInfoPlaceholder')}</p>
                        </div>
                    </div>
                </div>

                {/* Title */}
                <div className="text-center mb-6">
                    <h1 className="text-2xl font-bold text-gray-800 mb-2">{t('report.diagnosticReportTitle').toUpperCase()}</h1>
                    <div className="text-sm text-gray-600">{t('report.imagingDepartment').toUpperCase()}</div>
                </div>

                {/* Patient Information */}
                <div className="mb-6">
                    <h2 className="text-lg font-semibold text-gray-800 mb-3 pb-2 border-b border-gray-300">
                        I. {t('report.patientInformation').toUpperCase()}
                    </h2>
                    <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
                        <div className="flex">
                            <span className="font-semibold w-32">{t('report.fullName')}:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.ho_ten}</span>
                        </div>
                        <div className="flex">
                            <span className="font-semibold w-32">{t('report.age')}:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.tuoi}</span>
                        </div>
                        <div className="flex">
                            <span className="font-semibold w-32">{t('report.gender')}:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.gioi_tinh}</span>
                        </div>
                        <div className="flex">
                            <span className="font-semibold w-32">{t('report.bloodType')}:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.nhom_mau}</span>
                        </div>
                        <div className="flex">
                            <span className="font-semibold w-32">{t('report.scanDate')}:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.ngay_chup}</span>
                        </div>
                        <div className="flex">
                            <span className="font-semibold w-32">{t('report.readDate')}:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.ngay_doc_phim}</span>
                        </div>
                        <div className="flex col-span-2">
                            <span className="font-semibold w-32">{t('report.clinicalDiagnosis')}:</span>
                            <span className="flex-1">{reportData.thong_tin_benh_nhan.chan_doan_lam_sang}</span>
                        </div>
                    </div>
                </div>

                {/* Report Details */}
                <div className="mb-6">
                    <h2 className="text-lg font-semibold text-gray-800 mb-3 pb-2 border-b border-gray-300">
                        II. {t('report.xrayReport').toUpperCase()}
                    </h2>

                    <div className="space-y-3 text-sm">
                        <div>
                            <span className="font-semibold">{t('report.meshTags')}: </span>
                            <span className="text-gray-700">
                                {allXaiImages.map((xaiImg, index) => (
                                    <span key={index}>
                                        <a
                                            href={`#xai-${index}`}
                                            className="text-teal-600 hover:text-teal-700 underline cursor-pointer"
                                        >
                                            {xaiImg.finding}
                                        </a>
                                        {index < allXaiImages.length - 1 && ', '}
                                    </span>
                                ))}
                            </span>
                        </div>

                        <div>
                            <span className="font-semibold">{t('report.imageType')}: </span>
                            <span className="text-gray-700">{reportData.bao_cao_x_quang.loai_anh}</span>
                        </div>

                        <div>
                            <span className="font-semibold">{t('report.indication')}: </span>
                            <span className="text-gray-700">{reportData.bao_cao_x_quang.chi_dinh}</span>
                        </div>

                        <div>
                            <span className="font-semibold">{t('report.comparison')}: </span>
                            <span className="text-gray-700">{reportData.bao_cao_x_quang.so_sanh}</span>
                        </div>
                    </div>
                </div>

                {/* Description */}
                <div className="mb-6">
                    <h3 className="text-base font-semibold text-gray-800 mb-2">{t('report.description')}:</h3>
                    <p className="text-sm text-gray-700 leading-relaxed text-justify">
                        {reportData.bao_cao_x_quang.mo_ta}
                    </p>
                </div>

                {/* Conclusion */}
                <div className="mb-8">
                    <h3 className="text-base font-semibold text-gray-800 mb-2">{t('report.conclusion')}:</h3>
                    <p className="text-sm text-gray-700 leading-relaxed text-justify">
                        {reportData.bao_cao_x_quang.ket_luan}
                    </p>
                </div>

                {/* Medical Images Section */}
                <div id="medical-images" className="mb-8 page-break-before">
                    <h2 className="text-lg font-semibold text-gray-800 mb-3 pb-2 border-b border-gray-300">
                        III. {t('report.diagnosticImages').toUpperCase()}
                    </h2>

                    {/* Original Image */}
                    <div className="mb-6">
                        <div className="border-2 border-amber-400 rounded-lg p-3">
                            <div className="text-center mb-2">
                                <span className="text-sm font-semibold text-amber-600">{t('report.originalXray')}</span>
                            </div>
                            <img
                                src={originalImage}
                                alt="Original X-Ray"
                                className="w-full h-auto object-contain border border-gray-200 rounded mx-auto"
                                style={{ maxHeight: '500px', maxWidth: '600px' }}
                            />
                        </div>
                    </div>

                    {/* All xAI Images */}
                    {allXaiImages.length > 0 && (
                        <div>
                            <h3 className="text-base font-semibold text-gray-700 mb-3">
                                {t('report.xaiImages')} - {allXaiImages.length} {t('report.symptoms')}
                            </h3>
                            <div className="grid grid-cols-2 gap-4">
                                {allXaiImages.map((xaiImg, index) => (
                                    <div key={index} id={`xai-${index}`} className="border-2 border-teal-400 rounded-lg p-3 scroll-mt-4">
                                        <div className="text-center mb-2">
                                            <span className="text-sm font-semibold text-teal-600">
                                                xAI: {xaiImg.finding}
                                            </span>
                                            <span className={`ml-2 text-xs px-2 py-0.5 rounded-full ${xaiImg.severity === 'Cao' ? 'bg-red-100 text-red-700' :
                                                    xaiImg.severity === 'Trung bình' ? 'bg-amber-100 text-amber-700' :
                                                        'bg-green-100 text-green-700'
                                                }`}>
                                                {xaiImg.severity}
                                            </span>
                                        </div>
                                        <img
                                            src={xaiImg.url}
                                            alt={`xAI: ${xaiImg.finding}`}
                                            className="w-full h-auto object-contain border border-gray-200 rounded"
                                            style={{ maxHeight: '350px' }}
                                        />
                                    </div>
                                ))}
                            </div>
                            <p className="text-xs text-gray-500 italic mt-3 text-center">
                                {t('report.xaiNote')}
                            </p>
                        </div>
                    )}
                </div>

                {/* Signature Section */}
                <div className="mt-12 pt-6">
                    <div className="grid grid-cols-2 gap-4">
                        {/* Left signature box */}
                        <div className="relative">
                            <div className="border-2 border-gray-400 rounded-lg p-8 relative h-32 flex items-center justify-center">
                                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-white px-3">
                                    <span className="text-xs font-semibold text-gray-600 uppercase">{t('report.reportRecipient')}</span>
                                </div>
                                <p className="text-sm text-gray-500 italic text-center">{t('report.recipientSignature')}</p>
                            </div>
                        </div>

                        {/* Right signature box */}
                        <div className="relative">
                            <div className="border-2 border-gray-400 rounded-lg p-8 relative h-32 flex items-center justify-center">
                                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-white px-3">
                                    <span className="text-xs font-semibold text-gray-600 uppercase">{t('report.diagnosticDoctor')}</span>
                                </div>
                                <p className="text-sm text-gray-500 italic text-center">{t('report.doctorSignature')}</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="mt-6 pt-4 border-t border-gray-200 text-center text-xs text-gray-500">
                    <p>{t('report.autoGenerated')}</p>
                    <p>{t('report.contactDoctor')}</p>
                </div>
            </div>

            {/* Print styles */}
            <style jsx>{`
                html {
                    scroll-behavior: smooth;
                }
                
                .page-break-before {
                    page-break-before: auto;
                }
                
                @media print {
                    @page {
                        size: A4;
                        margin: 15mm;
                    }
                    
                    body {
                        print-color-adjust: exact;
                        -webkit-print-color-adjust: exact;
                    }
                    
                    .print\\:hidden {
                        display: none !important;
                    }
                    
                    .page-break-before {
                        page-break-before: always;
                    }
                }
            `}</style>
        </div>
    );
};
