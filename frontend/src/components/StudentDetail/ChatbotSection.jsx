import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';

export const ChatbotSection = () => {
    const [messages, setMessages] = useState([
        {
            id: 1,
            type: 'bot',
            text: "Xin chào! Tôi là trợ lý AI y khoa của bạn. Tôi có thể giúp bạn hiểu rõ hơn về ca bệnh này. Hãy hỏi tôi bất cứ điều gì!",
            timestamp: new Date()
        }
    ]);
    const [inputMessage, setInputMessage] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSendMessage = async () => {
        if (!inputMessage.trim()) return;

        // Add user message
        const userMessage = {
            id: Date.now(),
            type: 'user',
            text: inputMessage,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputMessage('');
        setIsTyping(true);

        // Simulate AI response
        setTimeout(() => {
            const botMessage = {
                id: Date.now() + 1,
                type: 'bot',
                text: generateMockResponse(inputMessage),
                timestamp: new Date()
            };
            setMessages(prev => [...prev, botMessage]);
            setIsTyping(false);
        }, 1500);
    };

    const generateMockResponse = (question) => {
        const responses = [
            "Dựa trên các mẫu hình hình ảnh, tôi có thể thấy các dấu hiệu phù hợp với chẩn đoán. Các chỉ số chính bao gồm...",
            "Đó là một câu hỏi hay! Trong những trường hợp như thế này, chúng ta thường tìm kiếm các dấu hiệu cụ thể trong kết quả chụp...",
            "Bệnh lý thể hiện ở đây được đặc trưng bởi một số đặc điểm nổi bật. Hãy để tôi giải thích...",
            "Quan sát tốt! Phát hiện này có ý nghĩa quan trọng vì nó gợi ý...",
            "Để trả lời câu hỏi của bạn, chúng ta cần xem xét nhiều yếu tố bao gồm tiền sử bệnh nhân và kết quả hình ảnh..."
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    return (
        <div className="h-full bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden flex flex-col">
            {/* Header */}
            <div className="px-4 py-3 border-b border-white/10 bg-[#141414] shrink-0">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-teal-500/20 rounded-lg flex items-center justify-center">
                        <Bot className="w-5 h-5 text-teal-500" />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-white">Trợ Lý AI</h3>
                        <p className="text-xs text-gray-400">Hỏi tôi bất cứ điều gì về ca bệnh này</p>
                    </div>
                </div>
            </div>

            {/* Messages Container */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
                    >
                        {/* Avatar */}
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 bg-teal-500/20 text-teal-500`}>
                            {message.type === 'bot' ? (
                                <Bot className="w-4 h-4" />
                            ) : (
                                <User className="w-4 h-4" />
                            )}
                        </div>

                        {/* Message Bubble */}
                        <div className={`flex-1 max-w-[80%] ${message.type === 'user' ? 'text-right' : ''}`}>
                            <div className={`inline-block px-4 py-2 rounded-lg text-sm ${message.type === 'bot'
                                ? 'bg-white/5 text-gray-200 border border-white/10'
                                : 'bg-teal-500 text-white'
                                }`}>
                                {message.text}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </div>
                        </div>
                    </div>
                ))}

                {/* Typing Indicator */}
                {isTyping && (
                    <div className="flex gap-3">
                        <div className="w-8 h-8 bg-teal-500/20 rounded-lg flex items-center justify-center shrink-0">
                            <Bot className="w-4 h-4 text-teal-500" />
                        </div>
                        <div className="bg-white/5 border border-white/10 px-4 py-2 rounded-lg">
                            <Loader2 className="w-4 h-4 text-gray-400 animate-spin" />
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="px-4 py-3 border-t border-white/10 bg-[#141414] shrink-0">
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Hỏi về ca bệnh này..."
                        className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-teal-500 transition-colors"
                    />
                    <button
                        onClick={handleSendMessage}
                        disabled={!inputMessage.trim() || isTyping}
                        className="px-4 py-2 bg-teal-500 hover:bg-teal-600 disabled:bg-white/5 disabled:text-gray-500 text-white rounded-lg transition-colors shrink-0"
                    >
                        <Send className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Custom Scrollbar Styles */}
            <style jsx>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 6px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 3px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background: rgba(20, 184, 166, 0.3);
                    border-radius: 3px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background: rgba(20, 184, 166, 0.5);
                }
            `}</style>
        </div>
    );
};
