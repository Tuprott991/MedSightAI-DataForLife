import { Link } from 'react-router-dom';

export const CTASection = () => {
    return (
        <div className="container mx-auto px-6 py-20">
            <div className="bg-linear-to-r from-teal-500/20 to-teal-600/20 border border-teal-500/30 rounded-2xl p-12 text-center">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">Sẵn Sàng Chuyển Đổi Chăm Sóc Sức Khỏe?</h2>
                <p className="text-gray-300 text-lg mb-8 max-w-2xl mx-auto">
                    Tham gia cùng hàng nghìn chuyên gia y tế đang sử dụng MedSightAI để cải thiện kết quả điều trị bệnh nhân và thúc đẩy kiến thức y khoa.
                </p>
                <div className="flex flex-wrap gap-4 justify-center">
                    <Link
                        to="/doctor"
                        className="inline-flex items-center gap-2 bg-teal-500 hover:bg-teal-600 text-white px-8 py-4 rounded-lg font-semibold transition-all transform hover:scale-105"
                    >
                        Bắt Đầu Với Tư Cách Bác Sĩ
                    </Link>
                    <Link
                        to="/student"
                        className="inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-8 py-4 rounded-lg font-semibold transition-all border border-white/20"
                    >
                        Bắt Đầu Học Tập
                    </Link>
                </div>
            </div>
        </div>
    );
};
