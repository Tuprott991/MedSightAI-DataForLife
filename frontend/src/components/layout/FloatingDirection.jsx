import { useState, useEffect } from "react";
import { ArrowUp } from "lucide-react";

export const FloatingDirection = () => {
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
        const toggleVisibility = () => {
            // Hiển thị button khi scroll xuống hơn 300px
            if (window.scrollY > 300) {
                setIsVisible(true);
            } else {
                setIsVisible(false);
            }
        };

        window.addEventListener("scroll", toggleVisibility);

        return () => {
            window.removeEventListener("scroll", toggleVisibility);
        };
    }, []);

    const handleScrollTop = () => {
        window.scrollTo({
            top: 0,
            behavior: "smooth",
        });
    };

    // Luôn render button nhưng với opacity và transform
    return (
        <button
            onClick={handleScrollTop}
            className={`fixed bottom-6 right-6 bg-teal-500 text-white px-4 py-3 rounded-lg shadow-lg hover:bg-teal-600 cursor-pointer transition-all duration-300 hover:scale-105 flex flex-row items-center gap-2 group z-50 ${isVisible
                ? 'opacity-100 translate-y-0'
                : 'opacity-0 translate-y-4 pointer-events-none'
                }`}
        >
            <ArrowUp className="w-5 h-5" />
            <span className="text-sm font-medium whitespace-nowrap">Lên đầu trang</span>
        </button>
    );
};
