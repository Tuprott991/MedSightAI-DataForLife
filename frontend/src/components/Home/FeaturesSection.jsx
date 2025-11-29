import { useState, useEffect, useRef } from 'react';
import { Brain, Heart, Shield, GraduationCap, Activity, Stethoscope } from 'lucide-react';
import { FeatureCard } from './FeatureCard';

const features = [
    {
        icon: Brain,
        title: 'Chẩn Đoán Bằng AI',
        description: 'Tận dụng thuật toán học máy tiên tiến để chẩn đoán y khoa chính xác và đề xuất phương pháp điều trị.'
    },
    {
        icon: Heart,
        title: 'Quản Lý Chăm Sóc Bệnh Nhân',
        description: 'Hồ sơ bệnh nhân toàn diện, lịch sử điều trị và đặt lịch hẹn trong một nền tảng thống nhất.'
    },
    {
        icon: Shield,
        title: 'Bảo Mật & Tuân Thủ',
        description: 'Bảo mật cấp doanh nghiệp với sự tuân thủ đầy đủ các tiêu chuẩn bảo vệ dữ liệu y tế và quy định.'
    },
    {
        icon: GraduationCap,
        title: 'Giáo Dục Y Khoa',
        description: 'Mô-đun học tập tương tác, mô phỏng ảo và tài nguyên y khoa đầy đủ cho sinh viên.'
    },
    {
        icon: Activity,
        title: 'Phân Tích Thời Gian Thực',
        description: 'Giám sát dấu hiệu sinh tồn bệnh nhân, theo dõi tiến trình điều trị và phân tích xu hướng sức khỏe với trực quan hóa dữ liệu thời gian thực.'
    },
    {
        icon: Stethoscope,
        title: 'Hợp Tác Chuyên Gia',
        description: 'Kết nối với các chuyên gia y tế trên toàn thế giới, chia sẻ kiến thức và hợp tác về các ca bệnh phức tạp.'
    }
];

export const FeaturesSection = () => {
    const [visibleCards, setVisibleCards] = useState([]);
    const sectionRef = useRef(null);

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        features.forEach((_, index) => {
                            setTimeout(() => {
                                setVisibleCards((prev) => {
                                    if (!prev.includes(index)) {
                                        return [...prev, index];
                                    }
                                    return prev;
                                });
                            }, index * 150); // Delay 150ms giữa mỗi card
                        });
                    }
                });
            },
            {
                threshold: 0.1,
                rootMargin: '0px'
            }
        );

        if (sectionRef.current) {
            observer.observe(sectionRef.current);
        }

        return () => {
            if (sectionRef.current) {
                observer.unobserve(sectionRef.current);
            }
        };
    }, []);

    return (
        <div ref={sectionRef} className="container mx-auto px-6 py-20">
            <div className="text-center mb-16">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">Tại Sao Chọn MedSightAI?</h2>
                <p className="text-gray-400 text-lg">Các tính năng tiên tiến được thiết kế cho chăm sóc sức khỏe hiện đại</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {features.map((feature, index) => (
                    <FeatureCard
                        key={index}
                        icon={feature.icon}
                        title={feature.title}
                        description={feature.description}
                        isVisible={visibleCards.includes(index)}
                        delay={index * 150}
                    />
                ))}
            </div>
        </div>
    );
};
