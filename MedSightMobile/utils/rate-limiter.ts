/**
 * Rate Limiter để bảo vệ chống spam OTP
 * Giới hạn số lần gửi OTP cho mỗi số điện thoại
 */

interface RateLimitRecord {
  count: number;
  firstAttempt: number;
  lastAttempt: number;
}

// In-memory storage (production nên dùng AsyncStorage hoặc backend)
const rateLimitStore: Record<string, RateLimitRecord> = {};

/**
 * Kiểm tra xem có thể gửi OTP cho số này không
 * @param phoneNumber - Số điện thoại
 * @param maxAttempts - Số lần tối đa trong khoảng thời gian
 * @param timeWindow - Khoảng thời gian (ms) - default: 1 giờ
 * @returns true nếu được phép gửi
 */
export const canSendOTP = (
  phoneNumber: string,
  maxAttempts: number = 5,
  timeWindow: number = 60 * 60 * 1000 // 1 giờ
): boolean => {
  const now = Date.now();
  const record = rateLimitStore[phoneNumber];

  // Lần đầu tiên gửi
  if (!record) {
    rateLimitStore[phoneNumber] = {
      count: 1,
      firstAttempt: now,
      lastAttempt: now,
    };
    return true;
  }

  // Nếu đã quá time window, reset counter
  if (now - record.firstAttempt > timeWindow) {
    rateLimitStore[phoneNumber] = {
      count: 1,
      firstAttempt: now,
      lastAttempt: now,
    };
    return true;
  }

  // Kiểm tra số lần gửi
  if (record.count >= maxAttempts) {
    const timeLeft = Math.ceil((timeWindow - (now - record.firstAttempt)) / 1000 / 60);
    console.warn(`⚠️ Rate limit exceeded for ${phoneNumber}. Try again in ${timeLeft} minutes.`);
    return false;
  }

  // Tăng counter
  record.count++;
  record.lastAttempt = now;
  return true;
};

/**
 * Lấy thông tin rate limit cho số điện thoại
 */
export const getRateLimitInfo = (phoneNumber: string): {
  remainingAttempts: number;
  timeUntilReset: number;
  isBlocked: boolean;
} => {
  const record = rateLimitStore[phoneNumber];
  const maxAttempts = 5;
  const timeWindow = 60 * 60 * 1000;

  if (!record) {
    return {
      remainingAttempts: maxAttempts,
      timeUntilReset: 0,
      isBlocked: false,
    };
  }

  const now = Date.now();
  const timeElapsed = now - record.firstAttempt;
  const timeLeft = Math.max(0, timeWindow - timeElapsed);

  return {
    remainingAttempts: Math.max(0, maxAttempts - record.count),
    timeUntilReset: timeLeft,
    isBlocked: record.count >= maxAttempts && timeLeft > 0,
  };
};

/**
 * Reset rate limit cho số điện thoại (dùng khi verify thành công)
 */
export const resetRateLimit = (phoneNumber: string): void => {
  delete rateLimitStore[phoneNumber];
};

/**
 * Format thời gian còn lại
 */
export const formatTimeLeft = (ms: number): string => {
  const minutes = Math.floor(ms / 1000 / 60);
  const seconds = Math.floor((ms / 1000) % 60);
  
  if (minutes > 0) {
    return `${minutes} phút ${seconds} giây`;
  }
  return `${seconds} giây`;
};

// Ví dụ sử dụng:
/*
import { canSendOTP, getRateLimitInfo, resetRateLimit } from '@/utils/rate-limiter';

// Trước khi gửi OTP
const handleSendOTP = async () => {
  if (!canSendOTP(phoneNumber, 5, 60 * 60 * 1000)) {
    const info = getRateLimitInfo(phoneNumber);
    Alert.alert(
      'Đã gửi quá nhiều lần',
      `Vui lòng thử lại sau ${formatTimeLeft(info.timeUntilReset)}`
    );
    return;
  }
  
  await sendOTP(phoneNumber, verifier);
};

// Sau khi verify thành công
const handleVerifyOTP = async () => {
  await verifyOTP(otp);
  resetRateLimit(phoneNumber); // Clear rate limit
};
*/
