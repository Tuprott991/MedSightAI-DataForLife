// OTP Enhancement Utilities
// Add these to improve the OTP implementation

import { useState, useEffect } from 'react';

/**
 * Hook for OTP countdown timer (rate limiting)
 * Usage: const { countdown, canResend, startCountdown } = useOTPTimer(60);
 */
export const useOTPTimer = (initialSeconds: number = 60) => {
  const [countdown, setCountdown] = useState(0);
  const [canResend, setCanResend] = useState(true);

  useEffect(() => {
    if (countdown <= 0) {
      setCanResend(true);
      return;
    }

    const timer = setInterval(() => {
      setCountdown(prev => prev - 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [countdown]);

  const startCountdown = () => {
    setCanResend(false);
    setCountdown(initialSeconds);
  };

  return { countdown, canResend, startCountdown };
};

/**
 * Format phone number for display
 * +84945876079 → +84 945 876 079
 */
export const formatPhoneDisplay = (phone: string): string => {
  const cleaned = phone.replace(/\s/g, '');
  
  // Vietnam format: +84 xxx xxx xxx
  if (cleaned.startsWith('+84')) {
    return cleaned.replace(/(\+84)(\d{3})(\d{3})(\d{3})/, '$1 $2 $3 $4');
  }
  
  // US format: +1 xxx xxx xxxx
  if (cleaned.startsWith('+1')) {
    return cleaned.replace(/(\+1)(\d{3})(\d{3})(\d{4})/, '$1 $2 $3 $4');
  }
  
  return phone;
};

/**
 * Vietnamese error messages for Firebase auth codes
 */
export const getVietnameseAuthError = (code: string): string => {
  const errors: Record<string, string> = {
    'auth/invalid-phone-number': 'Số điện thoại không hợp lệ',
    'auth/too-many-requests': 'Quá nhiều yêu cầu. Vui lòng thử lại sau.',
    'auth/invalid-verification-code': 'Mã OTP không đúng. Vui lòng kiểm tra lại.',
    'auth/code-expired': 'Mã OTP đã hết hạn. Vui lòng gửi mã mới.',
    'auth/network-request-failed': 'Lỗi kết nối mạng. Vui lòng kiểm tra internet.',
    'auth/operation-not-supported-in-this-environment': 'Môi trường không hỗ trợ xác thực số điện thoại',
    'auth/invalid-app-credential': 'ReCAPTCHA không hợp lệ. Vui lòng thử lại.',
    'auth/missing-phone-number': 'Vui lòng nhập số điện thoại',
    'auth/quota-exceeded': 'Đã vượt quá giới hạn. Vui lòng thử lại sau.',
    'auth/captcha-check-failed': 'Xác minh ReCAPTCHA thất bại',
    'auth/user-disabled': 'Tài khoản đã bị vô hiệu hóa',
  };
  
  return errors[code] || 'Đã có lỗi xảy ra. Vui lòng thử lại.';
};

/**
 * Format countdown time (seconds to MM:SS)
 */
export const formatCountdown = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

/**
 * Validate phone number with multiple formats
 */
export const validatePhoneNumber = (phone: string): boolean => {
  const cleaned = phone.replace(/\s/g, '');
  
  // Vietnam: 0xxxxxxxxx or +84xxxxxxxxx
  const vietnamPattern = /^(\+84|0)[0-9]{9,10}$/;
  
  // International: +xx xxxxxxxxxx
  const intlPattern = /^\+[1-9]\d{10,14}$/;
  
  return vietnamPattern.test(cleaned) || intlPattern.test(cleaned);
};

/**
 * Normalize phone number to E.164 format
 */
export const normalizePhoneNumber = (phone: string): string => {
  let cleaned = phone.replace(/\s/g, '').replace(/-/g, '');
  
  // Vietnam: convert 0xxx to +84xxx
  if (cleaned.startsWith('0')) {
    cleaned = '+84' + cleaned.slice(1);
  }
  
  // Ensure it starts with +
  if (!cleaned.startsWith('+')) {
    cleaned = '+' + cleaned;
  }
  
  return cleaned;
};

/**
 * Check if running on real device (not simulator/emulator)
 */
export const isRealDevice = (): boolean => {
  // This requires expo-device package
  // import * as Device from 'expo-device';
  // return Device.isDevice;
  
  // Fallback implementation
  return true; // Assume real device for now
};

/**
 * Analytics event tracking (placeholder)
 */
export const trackOTPEvent = (event: string, data?: Record<string, any>) => {
  console.log(`[Analytics] ${event}`, data);
  // TODO: Integrate with Firebase Analytics or other service
  // import analytics from '@react-native-firebase/analytics';
  // analytics().logEvent(event, data);
};

/**
 * OTP Input validation
 */
export const isValidOTP = (otp: string[]): boolean => {
  return otp.every(digit => digit.length === 1 && /^\d$/.test(digit));
};

/**
 * Debounce function for preventing spam clicks
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout | null = null;
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

/**
 * Throttle function for rate limiting
 */
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean = false;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};

/**
 * Safe async error wrapper
 */
export const safeAsync = async <T>(
  promise: Promise<T>,
  errorMessage?: string
): Promise<[T | null, Error | null]> => {
  try {
    const data = await promise;
    return [data, null];
  } catch (error) {
    console.error(errorMessage || 'Async operation failed:', error);
    return [null, error as Error];
  }
};

// Example usage in LoginScreen:
/*
import { useOTPTimer, formatPhoneDisplay, getVietnameseAuthError, trackOTPEvent } from './otp-utils';

const { countdown, canResend, startCountdown } = useOTPTimer(60);

const handleSendOTP = async () => {
  try {
    trackOTPEvent('otp_send_initiated', { phoneNumber });
    await sendOTP(phoneNumber, verifier);
    startCountdown(); // Start 60s cooldown
    trackOTPEvent('otp_send_success');
  } catch (err) {
    const errorMsg = getVietnameseAuthError(err.code);
    setError(errorMsg);
    trackOTPEvent('otp_send_failed', { error: err.code });
  }
};

// In UI:
<Text>Gửi mã đến {formatPhoneDisplay(phoneNumber)}</Text>

<Button
  onPress={handleSendOTP}
  disabled={!canResend || isLoading}
>
  {canResend ? 'Gửi lại' : `Gửi lại sau ${countdown}s`}
</Button>
*/
