export const Measurements = ({ metrics }) => {
    return (
        <div>
            <h3 className="text-sm font-semibold text-white mb-3">Measurements</h3>
            <div className="grid grid-cols-2 gap-2">
                {metrics.map((metric, index) => (
                    <div key={index} className="p-2.5 rounded-lg border border-white/10 bg-[#0f0f0f]">
                        <p className="text-xs text-gray-500 mb-1">{metric.label}</p>
                        <p className="text-sm font-semibold text-white">{metric.value}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};
