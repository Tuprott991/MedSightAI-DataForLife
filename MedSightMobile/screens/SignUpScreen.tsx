import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  TouchableOpacity,
  TextInput as RNTextInput,
} from 'react-native';
import { TextInput, Button, Text, ActivityIndicator, HelperText } from 'react-native-paper';
import { useAuth } from '@/context/AuthContext';

export default function SignUpScreen({ navigation }: any) {
  const { sendOTP, verifyOTP, isLoading, error, setError } = useAuth();
  
  const [phoneNumber, setPhoneNumber] = useState('');
  const [step, setStep] = useState<'phone' | 'otp'>('phone');
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [isPhoneValid, setIsPhoneValid] = useState(false);
  const otpInputRefs = useRef<RNTextInput[]>([]);

  // Validate phone number
  const validatePhone = (phone: string) => {
    const phoneRegex = /^(\+\d{1,3}|0)\d{9,10}$/;
    return phoneRegex.test(phone.replace(/\s/g, ''));
  };

  const handlePhoneChange = (text: string) => {
    setPhoneNumber(text);
    setIsPhoneValid(validatePhone(text));
    if (error) setError(null);
  };

  const handleSendOTP = async () => {
    try {
      setError(null);
      await sendOTP(phoneNumber);
      setStep('otp');
    } catch (err) {
      console.error('Error sending OTP:', err);
    }
  };

  const handleOtpChange = (text: string, index: number) => {
    if (text.length <= 1 && /^\d*$/.test(text)) {
      const newOtp = [...otp];
      newOtp[index] = text;
      setOtp(newOtp);

      // Auto-focus to next field
      if (text && index < 5) {
        otpInputRefs.current[index + 1]?.focus();
      }

      if (error) setError(null);
    }
  };

  const handleOtpKeyPress = (key: string, index: number) => {
    if (key === 'Backspace' && !otp[index] && index > 0) {
      otpInputRefs.current[index - 1]?.focus();
      const newOtp = [...otp];
      newOtp[index - 1] = '';
      setOtp(newOtp);
    }
  };

  const handleVerifyOTP = async () => {
    try {
      setError(null);
      const otpCode = otp.join('');
      if (otpCode.length !== 6) {
        setError('Please enter 6-digit OTP');
        return;
      }
      await verifyOTP(otpCode);
      // Navigation will be handled by the auth state change
    } catch (err) {
      console.error('Error verifying OTP:', err);
    }
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <Text variant="displaySmall" style={styles.title}>
            MedSight
          </Text>
          <Text variant="bodyLarge" style={styles.subtitle}>
            {step === 'phone' ? 'Đăng ký tài khoản' : 'Xác minh OTP'}
          </Text>
        </View>

        {/* Content */}
        <View style={styles.content}>
          {step === 'phone' ? (
            <>
              {/* Phone Number Input */}
              <Text variant="labelLarge" style={styles.label}>
                Số điện thoại
              </Text>
              <TextInput
                label="Nhập số điện thoại (+84...)"
                value={phoneNumber}
                onChangeText={handlePhoneChange}
                keyboardType="phone-pad"
                mode="outlined"
                style={styles.input}
                placeholder="+84 9xx xxx xxx"
                editable={!isLoading}
                outlineColor="#14B8A6"
                activeOutlineColor="#0D9488"
              />
              {phoneNumber && !isPhoneValid && (
                <HelperText type="error" visible={true}>
                  Số điện thoại không hợp lệ
                </HelperText>
              )}
              <Text variant="bodySmall" style={styles.infoText}>
                Chúng tôi sẽ gửi mã xác minh OTP đến số điện thoại này
              </Text>

              {/* Error Message */}
              {error && (
                <View style={styles.errorBox}>
                  <Text style={styles.errorText}>{error}</Text>
                </View>
              )}

              {/* Send OTP Button */}
              <Button
                mode="contained"
                onPress={handleSendOTP}
                disabled={!isPhoneValid || isLoading}
                loading={isLoading}
                style={styles.button}
                buttonColor="#14B8A6"
              >
                {isLoading ? 'Đang gửi...' : 'Gửi mã OTP'}
              </Button>

              {/* Sign In Link */}
              <View style={styles.footer}>
                <Text variant="bodyMedium">Đã có tài khoản? </Text>
                <TouchableOpacity onPress={() => navigation?.navigate('login')}>
                  <Text
                    variant="bodyMedium"
                    style={styles.linkText}
                  >
                    Đăng nhập
                  </Text>
                </TouchableOpacity>
              </View>
            </>
          ) : (
            <>
              {/* OTP Input */}
              <Text variant="labelLarge" style={styles.label}>
                Mã xác minh (6 chữ số)
              </Text>
              <Text variant="bodySmall" style={styles.infoText}>
                Nhập mã xác minh được gửi đến {phoneNumber}
              </Text>

              <View style={styles.otpContainer}>
                {otp.map((digit, index) => (
                  <RNTextInput
                    key={index}
                    ref={(ref) => {
                      if (ref) otpInputRefs.current[index] = ref;
                    }}
                    style={styles.otpInput}
                    maxLength={1}
                    keyboardType="number-pad"
                    value={digit}
                    onChangeText={(text) => handleOtpChange(text, index)}
                    onKeyPress={({ nativeEvent }) =>
                      handleOtpKeyPress(nativeEvent.key, index)
                    }
                    editable={!isLoading}
                    placeholderTextColor="#CCC"
                  />
                ))}
              </View>

              {/* Error Message */}
              {error && (
                <View style={styles.errorBox}>
                  <Text style={styles.errorText}>{error}</Text>
                </View>
              )}

              {/* Verify Button */}
              <Button
                mode="contained"
                onPress={handleVerifyOTP}
                disabled={otp.some((o) => !o) || isLoading}
                loading={isLoading}
                style={styles.button}
                buttonColor="#14B8A6"
              >
                {isLoading ? 'Đang xác minh...' : 'Xác minh'}
              </Button>

              {/* Back Button */}
              <Button
                mode="outlined"
                onPress={() => {
                  setStep('phone');
                  setOtp(['', '', '', '', '', '']);
                  setError(null);
                }}
                style={styles.backButton}
                textColor="#14B8A6"
                disabled={isLoading}
              >
                Quay lại
              </Button>

              {/* Resend OTP */}
              <View style={styles.footer}>
                <Text variant="bodySmall">Không nhận được mã? </Text>
                <TouchableOpacity
                  onPress={handleSendOTP}
                  disabled={isLoading}
                >
                  <Text
                    variant="bodySmall"
                    style={styles.linkText}
                  >
                    Gửi lại
                  </Text>
                </TouchableOpacity>
              </View>
            </>
          )}
        </View>

        {/* Loading Indicator */}
        {isLoading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator animating={true} size="large" color="#14B8A6" />
          </View>
        )}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#151718',
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 20,
  },
  header: {
    marginBottom: 40,
    alignItems: 'center',
  },
  title: {
    fontWeight: 'bold',
    color: '#14B8A6',
    marginBottom: 8,
  },
  subtitle: {
    color: '#ECEDEE',
    textAlign: 'center',
  },
  content: {
    marginBottom: 20,
  },
  label: {
    color: '#ECEDEE',
    marginBottom: 8,
    fontWeight: '600',
  },
  input: {
    marginBottom: 12,
    backgroundColor: '#1F2937',
    color: '#ECEDEE',
  },
  infoText: {
    color: '#9BA1A6',
    marginBottom: 16,
    marginTop: -8,
  },
  button: {
    marginVertical: 16,
    paddingVertical: 8,
  },
  backButton: {
    marginVertical: 8,
    borderColor: '#14B8A6',
  },
  otpContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 24,
  },
  otpInput: {
    width: 48,
    height: 56,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#14B8A6',
    backgroundColor: '#1F2937',
    color: '#ECEDEE',
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 24,
  },
  linkText: {
    color: '#14B8A6',
    fontWeight: '600',
    textDecorationLine: 'underline',
  },
  errorBox: {
    backgroundColor: '#DC2626',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    marginBottom: 16,
  },
  errorText: {
    color: '#FFFFFF',
    fontSize: 14,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
});
