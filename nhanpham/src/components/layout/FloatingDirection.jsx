import { ArrowUp } from "lucide-react";

export const FloatingDirection = () => {
    const handleScrollTop = () => {
        window.scrollTo({
            top: 0,
            behavior: "smooth",
        });
    };

    return (
        <button
            onClick={handleScrollTop}
            className="fixed bottom-6 right-6 bg-teal-500 text-white px-2 py-2 rounded-full shadow-lg hover:bg-emerald-600 cursor-pointer"
        >
            <ArrowUp />
        </button>
    );
};
