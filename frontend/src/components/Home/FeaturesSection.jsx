import { useState, useEffect, useRef } from 'react';
import { Brain, Heart, Shield, GraduationCap, Activity, Stethoscope } from 'lucide-react';
import { FeatureCard } from './FeatureCard';
import { useTranslation } from 'react-i18next';

export const FeaturesSection = () => {
    const { t } = useTranslation();
    const [visibleCards, setVisibleCards] = useState([]);
    const sectionRef = useRef(null);

    const features = [
        {
            icon: Brain,
            title: t('home.features.aiDiagnostics.title'),
            description: t('home.features.aiDiagnostics.description')
        },
        {
            icon: Heart,
            title: t('home.features.similarCases.title'),
            description: t('home.features.similarCases.description')
        },
        {
            icon: Shield,
            title: t('home.features.secureCloud.title'),
            description: t('home.features.secureCloud.description')
        },
        {
            icon: GraduationCap,
            title: t('home.features.interactiveLearning.title'),
            description: t('home.features.interactiveLearning.description')
        },
        {
            icon: Activity,
            title: t('home.features.reportGeneration.title'),
            description: t('home.features.reportGeneration.description')
        },
        {
            icon: Stethoscope,
            title: t('home.features.realTimeCollab.title'),
            description: t('home.features.realTimeCollab.description')
        }
    ];

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
    }, [features]);

    return (
        <div ref={sectionRef} className="container mx-auto px-6 py-20">
            <div className="text-center mb-16">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">{t('home.features.title')}</h2>
                <p className="text-gray-400 text-lg">{t('home.features.subtitle')}</p>
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
