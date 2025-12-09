import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, MapPin, AlertTriangle, CheckCircle2, Info } from 'lucide-react';
import { useTranslation } from 'react-i18next';

// Component để format markdown đơn giản
const FormattedMessage = ({ text }) => {
    const formatText = (text) => {
        // Split by lines để giữ nguyên line breaks
        const lines = text.split('\n');

        return lines.map((line, lineIdx) => {
            // Xử lý **bold**
            const parts = line.split(/(\*\*.*?\*\*)/g);

            return (
                <span key={lineIdx}>
                    {parts.map((part, partIdx) => {
                        if (part.startsWith('**') && part.endsWith('**')) {
                            return <strong key={partIdx} className="font-semibold">{part.slice(2, -2)}</strong>;
                        }
                        return <span key={partIdx}>{part}</span>;
                    })}
                    {lineIdx < lines.length - 1 && <br />}
                </span>
            );
        });
    };

    return <>{formatText(text)}</>;
};

export const ChatbotSection = ({ annotations = [], caseData = null, submissionData = null }) => {
    const { t } = useTranslation();
    const [messages, setMessages] = useState([
        {
            id: 1,
            type: 'bot',
            text: t('studentDetail.chatbot.greeting'),
            timestamp: new Date()
        }
    ]);
    const [inputMessage, setInputMessage] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [streamingMessage, setStreamingMessage] = useState('');
    const messagesEndRef = useRef(null);
    const previousSubmissionData = useRef(null);

    // Mock ground truth data - vị trí đúng của bệnh lý
    const groundTruth = {
        regions: [
            { x: 250, y: 150, width: 180, height: 200, label: 'Đám mờ phổi', severity: 'high' },
            { x: 180, y: 320, width: 120, height: 140, label: 'Xơ hóa', severity: 'medium' }
        ],
        // Chỉ dùng 1 ảnh kết quả thực tế
        aiResultUrl: '/src/mock_data/patient_data/01_Tuberculosis/Consolidation/Untitled.jpeg'
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, streamingMessage]);

    // Theo dõi khi sinh viên submit chẩn đoán
    useEffect(() => {
        if (submissionData && submissionData !== previousSubmissionData.current) {
            previousSubmissionData.current = submissionData;
            analyzeSubmission(submissionData);
        }
    }, [submissionData]);

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

    // Phân tích submission của sinh viên
    const analyzeSubmission = (submission) => {
        setIsTyping(true);

        // Delay 2 giây trước khi bắt đầu phản hồi
        setTimeout(() => {
            const analysis = analyzeAnnotationsAndDiagnosis(submission);
            streamResponse(analysis);
        }, 2000);
    };

    // Stream response từng chữ một
    const streamResponse = async (analysis) => {
        const { feedbackMessage, messageType, images } = analysis;

        // Tạo message placeholder
        const messageId = Date.now();
        const newMessage = {
            id: messageId,
            type: 'bot',
            text: '',
            timestamp: new Date(),
            messageType: messageType,
            images: images,
            isStreaming: true
        };

        setMessages(prev => [...prev, newMessage]);
        setIsTyping(false);

        // Stream text
        let currentText = '';
        const words = feedbackMessage.split(' ');

        for (let i = 0; i < words.length; i++) {
            currentText += (i > 0 ? ' ' : '') + words[i];

            setMessages(prev => prev.map(msg =>
                msg.id === messageId
                    ? { ...msg, text: currentText }
                    : msg
            ));

            // Random delay giữa 30-80ms cho mỗi từ
            await new Promise(resolve => setTimeout(resolve, Math.random() * 50 + 30));
        }

        // Đánh dấu hoàn thành streaming
        setMessages(prev => prev.map(msg =>
            msg.id === messageId
                ? { ...msg, isStreaming: false }
                : msg
        ));
    };

    // Phân tích annotations và chẩn đoán
    const analyzeAnnotationsAndDiagnosis = (submission) => {
        const { diagnosis, annotations } = submission;
        const correctDiagnosis = caseData?.diagnosis || 'Lao phổi';

        let feedbackMessage = '';
        let messageType = 'info';
        let images = [];

        // Kiểm tra số lượng annotations
        const hasAnnotations = annotations && annotations.length > 0;

        if (!hasAnnotations) {
            messageType = 'warning';
            feedbackMessage = `Không phát hiện vùng đánh dấu tổn thương.\n\n`;
            feedbackMessage += `Chẩn đoán gửi lên: ${diagnosis}\n\n`;
            feedbackMessage += `Yêu cầu: Sử dụng công cụ vẽ để định vị chính xác các vùng bất thường trên phim chụp trước khi submit.\n\n`;
            feedbackMessage += `Kết quả phân tích thực tế:`;
            images = [
                { type: 'ai_result', url: groundTruth.aiResultUrl, label: 'Phân tích thực tế' }
            ];
            return { feedbackMessage, messageType, images };
        }

        // Phân tích từng annotation
        const annotationResults = annotations.map((ann, idx) =>
            checkAnnotationAccuracy(ann)
        );

        const correctCount = annotationResults.filter(r => r.accuracy === 'correct').length;
        const partialCount = annotationResults.filter(r => r.accuracy === 'partial').length;
        const incorrectCount = annotationResults.filter(r => r.accuracy === 'incorrect').length;

        // Kiểm tra chẩn đoán
        const diagnosisCorrect = diagnosis.toLowerCase().includes(correctDiagnosis.toLowerCase()) ||
            correctDiagnosis.toLowerCase().includes(diagnosis.toLowerCase());

        // Tạo phản hồi dựa trên kết quả
        if (correctCount === annotations.length && diagnosisCorrect) {
            // Trường hợp hoàn hảo: Cả chẩn đoán và vùng đều đúng
            messageType = 'success';
            feedbackMessage = `Đánh giá: Chẩn đoán chính xác\n\n`;
            feedbackMessage += `Chẩn đoán: ${diagnosis}\n`;
            feedbackMessage += `Độ chính xác vùng đánh dấu: ${correctCount}/${annotations.length}\n\n`;
            feedbackMessage += `Phân tích từng vùng:\n`;
            annotationResults.forEach((result, idx) => {
                if (result.accuracy === 'correct') {
                    feedbackMessage += `Vùng ${idx + 1}: ${result.matchedRegion.label} - Overlap ${result.overlap}%\n`;
                }
            });
            feedbackMessage += `\nNhận xét: ${getClinicSignificance(groundTruth.regions[0].label)}`;
        } else if (diagnosisCorrect && (correctCount === 0 && incorrectCount > 0)) {
            // Trường hợp: Chẩn đoán đúng nhưng vùng khoanh hoàn toàn sai
            messageType = 'error';
            feedbackMessage = `Đánh giá: Chẩn đoán đúng nhưng định vị tổn thương sai\n\n`;
            feedbackMessage += `Chẩn đoán: ${diagnosis} - Chính xác\n`;
            feedbackMessage += `Vùng đánh dấu: ${incorrectCount} vùng sai vị trí\n\n`;

            feedbackMessage += `Nhận xét:\n`;
            feedbackMessage += `Bạn đã xác định đúng bệnh lý nhưng chưa định vị chính xác vị trí tổn thương trên phim.\n\n`;

            feedbackMessage += `Phân tích chi tiết:\n`;
            annotationResults.forEach((result, idx) => {
                feedbackMessage += `Vùng ${idx + 1}: ${result.reason}\n`;
            });

            feedbackMessage += `\nHướng dẫn định vị tổn thương ${correctDiagnosis}:\n`;
            feedbackMessage += `- Vị trí: ${groundTruth.regions[0].label} thường ở thùy trên hoặc giữa phổi\n`;
            feedbackMessage += `- Đặc điểm: Vùng mật độ tăng, đậm hơn nền phổi bình thường\n`;
            feedbackMessage += `- Ranh giới: Không đều, có thể lan tỏa hoặc khu trú\n`;
            feedbackMessage += `- Kích thước: Thường từ 1-3cm, có thể lớn hơn\n\n`;
            feedbackMessage += `Kéo ảnh dưới đây sang khung hiển thị để học cách định vị chính xác:`;

            images = [
                { type: 'ai_result', url: groundTruth.aiResultUrl, label: 'Kết quả thực tế' }
            ];
        } else if (diagnosisCorrect && correctCount < annotations.length) {
            // Trường hợp: Chẩn đoán đúng nhưng một số vùng sai
            messageType = 'warning';
            feedbackMessage = `Đánh giá: Chẩn đoán đúng nhưng cần cải thiện định vị\n\n`;
            feedbackMessage += `Chẩn đoán: ${diagnosis} - Chính xác\n`;
            feedbackMessage += `Kết quả đánh dấu: ${correctCount} chính xác, ${partialCount} gần đúng, ${incorrectCount} sai\n\n`;

            feedbackMessage += `Nhận xét:\n`;
            feedbackMessage += `Bạn đã xác định đúng bệnh lý và một phần vị trí tổn thương. Cần cải thiện độ chính xác trong việc khoanh vùng.\n\n`;

            feedbackMessage += `Phân tích từng vùng:\n`;
            annotationResults.forEach((result, idx) => {
                if (result.accuracy === 'correct') {
                    feedbackMessage += `Vùng ${idx + 1}: Chính xác - ${result.matchedRegion.label}\n`;
                } else if (result.accuracy === 'partial') {
                    feedbackMessage += `Vùng ${idx + 1}: Overlap ${result.overlap}% - ${result.suggestion}\n`;
                } else {
                    feedbackMessage += `Vùng ${idx + 1}: Sai vị trí - ${result.reason}\n`;
                }
            });

            feedbackMessage += `\nYêu cầu:\n`;
            feedbackMessage += `- Quan sát kỹ hơn ranh giới tổn thương\n`;
            feedbackMessage += `- Mở rộng hoặc thu nhỏ vùng cho chính xác\n`;
            feedbackMessage += `- So sánh với kết quả thực tế phía dưới`;

            images = [
                { type: 'ai_result', url: groundTruth.aiResultUrl, label: 'Kết quả phân tích thực tế' }
            ];
        } else if (!diagnosisCorrect && correctCount === annotations.length) {
            // Trường hợp: Vùng đúng nhưng chẩn đoán sai
            messageType = 'warning';
            feedbackMessage = `Đánh giá: Định vị chính xác nhưng chẩn đoán sai\n\n`;
            feedbackMessage += `Chẩn đoán gửi lên: ${diagnosis}\n`;
            feedbackMessage += `Chẩn đoán chuẩn: ${correctDiagnosis}\n`;
            feedbackMessage += `Vùng đánh dấu: ${correctCount}/${annotations.length} chính xác\n\n`;

            feedbackMessage += `Nhận xét:\n`;
            feedbackMessage += `Bạn đã định vị chính xác vị trí tổn thương nhưng nhận định sai về bệnh lý.\n\n`;

            feedbackMessage += `Phân tích:\n`;
            annotationResults.forEach((result, idx) => {
                if (result.accuracy === 'correct') {
                    feedbackMessage += `Vùng ${idx + 1}: ${result.matchedRegion.label} - Overlap ${result.overlap}%\n`;
                }
            });

            feedbackMessage += `\nĐặc điểm phân biệt ${correctDiagnosis}:\n`;
            feedbackMessage += `${getClinicSignificance(groundTruth.regions[0].label)}\n\n`;
            feedbackMessage += `Yêu cầu xem xét lại tiền sử bệnh, triệu chứng lâm sàng và đặc điểm hình ảnh để đưa ra chẩn đoán chính xác.`;
        } else {
            // Trường hợp: Cả chẩn đoán và vùng đều sai
            messageType = 'error';
            feedbackMessage = `Đánh giá: Chẩn đoán chưa đạt\n\n`;
            feedbackMessage += `Chẩn đoán gửi lên: ${diagnosis}\n`;
            if (!diagnosisCorrect) {
                feedbackMessage += `Chẩn đoán chuẩn: ${correctDiagnosis}\n`;
            }
            feedbackMessage += `Kết quả đánh dấu: ${correctCount}/${annotations.length} chính xác\n\n`;

            feedbackMessage += `Phân tích lỗi:\n`;
            annotationResults.forEach((result, idx) => {
                if (result.accuracy === 'incorrect') {
                    feedbackMessage += `Vùng ${idx + 1}: ${result.reason}\n`;
                } else if (result.accuracy === 'partial') {
                    feedbackMessage += `Vùng ${idx + 1}: Gần đúng nhưng cần điều chỉnh\n`;
                }
            });

            feedbackMessage += `\nĐặc điểm ${groundTruth.regions[0].label} (${correctDiagnosis}):\n`;
            feedbackMessage += `- Vị trí: Thùy trên hoặc giữa phổi\n`;
            feedbackMessage += `- Hình ảnh: Mật độ tăng, vùng đậm hơn\n`;
            feedbackMessage += `- Ranh giới: Không đều, có thể kèm xơ hóa\n\n`;
            feedbackMessage += `Kéo ảnh dưới đây sang khung hiển thị để so sánh:`;

            images = [
                { type: 'ai_result', url: groundTruth.aiResultUrl, label: 'Kết quả thực tế' }
            ];
        }

        return { feedbackMessage, messageType, images };
    };

    // Kiểm tra độ chính xác của annotation
    const checkAnnotationAccuracy = (annotation) => {
        const { x, y, width, height } = annotation;

        // Kiểm tra overlap với ground truth regions
        for (const region of groundTruth.regions) {
            const overlap = calculateOverlap(
                { x, y, width, height },
                region
            );

            if (overlap > 70) {
                return {
                    accuracy: 'correct',
                    overlap: Math.round(overlap),
                    matchedRegion: region
                };
            } else if (overlap > 30) {
                return {
                    accuracy: 'partial',
                    overlap: Math.round(overlap),
                    matchedRegion: region,
                    suggestion: overlap < 50 ? 'mở rộng vùng' : 'thu nhỏ và dịch chuyển',
                    hint: `Tổn thương ${region.label} thường có mật độ đậm hơn và ranh giới rõ ràng hơn.`
                };
            }
        }

        // Kiểm tra vị trí chung
        const isInLungArea = y > 100 && y < 500 && x > 100 && x < 600;

        if (!isInLungArea) {
            return {
                accuracy: 'incorrect',
                reason: 'Vùng bạn đánh dấu nằm ngoài khu vực phổi.',
                guidance: '• Tập trung vào vùng giữa ngực\n• Tránh vùng tim (giữa dưới)\n• Tránh vùng xương sườn (hai bên)',
                observationTips: '• Quan sát độ đậm nhạt bất thường\n• Tìm các vùng mờ hoặc đông đặc\n• So sánh hai bên phổi'
            };
        }

        return {
            accuracy: 'incorrect',
            reason: 'Chưa phát hiện đúng vị trí tổn thương chính.',
            guidance: `• Tìm vùng có mật độ tăng (sáng hơn)\n• Chú ý các đám mờ ở thùy trên phổi\n• Quan sát sự khác biệt giữa 2 bên phổi`,
            observationTips: `• **${groundTruth.regions[0].label}:** Thường ở thùy trên hoặc giữa\n• Mật độ tăng, ranh giới không đều\n• Có thể kèm xơ hóa xung quanh`
        };
    };

    // Tính overlap giữa 2 bounding box
    const calculateOverlap = (box1, box2) => {
        const x1 = Math.max(box1.x, box2.x);
        const y1 = Math.max(box1.y, box2.y);
        const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
        const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

        if (x2 < x1 || y2 < y1) return 0;

        const intersectionArea = (x2 - x1) * (y2 - y1);
        const box1Area = box1.width * box1.height;
        const box2Area = box2.width * box2.height;
        const unionArea = box1Area + box2Area - intersectionArea;

        return (intersectionArea / unionArea) * 100;
    };

    // Lấy ý nghĩa lâm sàng
    const getClinicSignificance = (label) => {
        const significance = {
            'Đám mờ phổi': 'Tổn thương này gợi ý viêm hoặc lao phổi. Cần kết hợp xét nghiệm đờm và tiền sử ho kéo dài.',
            'Xơ hóa': 'Xơ hóa phổi là dấu hiệu của tổn thương mạn tính, thường gặp ở bệnh lao cũ đã điều trị.',
            'Tim to': 'Tăng kích thước tim có thể do suy tim hoặc tăng áp phổi kéo dài.'
        };
        return significance[label] || 'Tổn thương này cần đánh giá thêm với các xét nghiệm bổ sung.';
    };

    const generateMockResponse = (question) => {
        const lowerQuestion = question.toLowerCase();

        // Hỏi về vị trí tổn thương
        if (lowerQuestion.includes('ở đâu') || lowerQuestion.includes('vị trí') || lowerQuestion.includes('nằm')) {
            return `📍 **Vị trí tổn thương:**\n\nTrong ca bệnh này, các tổn thương chính nằm ở:\n• **Thùy trên phổi phải** - Đám mờ rõ ràng\n• **Vùng quanh rốn phổi** - Xơ hóa nhẹ\n\nBạn có thể thử khoanh vùng các khu vực bạn cho là bất thường, tôi sẽ đánh giá xem có chính xác không! 🎯`;
        }

        // Hỏi về heatmap hoặc AI
        if (lowerQuestion.includes('heatmap') || lowerQuestion.includes('ai phát hiện') || lowerQuestion.includes('máy nhận')) {
            return `🤖 **Phân tích AI:**\n\nAI đã phát hiện các vùng bất thường với độ tin cậy cao. Bạn muốn xem heatmap để so sánh với vùng bạn đã khoanh không?\n\nHeatmap sẽ hiển thị:\n🔴 Vùng đỏ: Bất thường mức cao\n🟡 Vùng vàng: Nghi ngờ\n🟢 Vùng xanh: Bình thường`;
        }

        // Hỏi về cách nhận biết
        if (lowerQuestion.includes('nhận biết') || lowerQuestion.includes('phát hiện') || lowerQuestion.includes('cách')) {
            return `🔍 **Cách nhận biết tổn thương:**\n\n1. **Quan sát mật độ:** Vùng bệnh thường sáng hơn (tăng đậm độ)\n2. **So sánh 2 bên:** Tìm sự khác biệt giữa phổi trái và phải\n3. **Ranh giới:** Tổn thương thường có ranh giới không rõ\n4. **Vị trí:** Lao phổi hay gặp ở thùy trên\n\nHãy thử khoanh vùng, tôi sẽ góp ý ngay! 💪`;
        }

        // Hỏi về chẩn đoán
        if (lowerQuestion.includes('chẩn đoán') || lowerQuestion.includes('bệnh gì')) {
            return `🏥 **Chẩn đoán:**\n\n${caseData?.diagnosis || 'Lao phổi'} - Độ tin cậy AI: 87%\n\n**Căn cứ chẩn đoán:**\n• Đám mờ ở thùy trên phổi\n• Có dấu hiệu xơ hóa\n• Ranh giới không đều\n\n**Cần làm thêm:**\n• Xét nghiệm đờm tìm BK\n• Test GeneXpert\n• CT scan nếu cần thiết`;
        }

        // Hỏi về điều trị
        if (lowerQuestion.includes('điều trị') || lowerQuestion.includes('thuốc')) {
            return `💊 **Phác đồ điều trị:**\n\n**Giai đoạn tấn công (2 tháng):**\n• Rifampicin + Isoniazid + Pyrazinamid + Ethambutol\n\n**Giai đoạn ổn định (4 tháng):**\n• Rifampicin + Isoniazid\n\n⚠️ **Lưu ý:**\n• Uống thuốc đều đặn\n• Không tự ý ngừng thuốc\n• Tái khám định kỳ`;
        }

        // Response mặc định
        return `Đó là câu hỏi hay! Bạn có thể:\n\n📝 Thử khoanh vùng các tổn thương trên ảnh\n🤖 Hỏi tôi về 'heatmap' để xem phân tích AI\n📍 Hỏi về 'vị trí' tổn thương\n🔍 Hỏi 'cách nhận biết' bệnh lý\n\nTôi sẽ đánh giá và góp ý cho bạn ngay! 💪`;
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
                        <h3 className="text-sm font-semibold text-white">{t('studentDetail.chatbot.title')}</h3>
                        <p className="text-xs text-gray-400">{t('studentDetail.chatbot.subtitle')}</p>
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
                                ? message.messageType === 'success' ? 'bg-green-500/10 text-green-200 border border-green-500/30'
                                    : message.messageType === 'warning' ? 'bg-yellow-500/10 text-yellow-200 border border-yellow-500/30'
                                        : message.messageType === 'error' ? 'bg-red-500/10 text-red-200 border border-red-500/30'
                                            : 'bg-white/5 text-gray-200 border border-white/10'
                                : 'bg-teal-500 text-white'
                                }`}>
                                <div className="whitespace-pre-line">
                                    <FormattedMessage text={message.text} />
                                    {message.isStreaming && <span className="animate-pulse">▋</span>}
                                </div>

                                {/* Hiển thị hình ảnh nếu có */}
                                {message.images && message.images.length > 0 && (
                                    <div className="mt-3 space-y-2">
                                        {message.images.map((img, idx) => (
                                            <div key={idx} className="border border-white/20 rounded-lg overflow-hidden">
                                                {img.label && (
                                                    <div className="bg-white/10 px-2 py-1 text-xs font-semibold">
                                                        {img.label}
                                                    </div>
                                                )}
                                                <img
                                                    src={img.url}
                                                    alt={img.label || 'Kết quả thực tế'}
                                                    draggable="true"
                                                    onDragStart={(e) => {
                                                        e.dataTransfer.setData('imageUrl', img.url);
                                                        e.dataTransfer.setData('imageLabel', img.label || 'Kết quả thực tế');
                                                    }}
                                                    className="w-full h-auto cursor-move hover:opacity-80 transition-opacity"
                                                    title="Kéo ảnh này sang khung hiển thị để so sánh"
                                                />
                                                <div className="bg-white/5 px-2 py-1 text-xs text-gray-400">
                                                    Kéo sang khung ảnh để so sánh
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
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
                        placeholder={t('studentDetail.chatbot.placeholder')}
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
