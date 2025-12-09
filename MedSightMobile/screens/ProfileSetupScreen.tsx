import React, { useState } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { TextInput, Button, Text, ActivityIndicator } from 'react-native-paper';
import { useRouter } from 'expo-router';
import { useAuth } from '@/context/AuthContext';

export default function ProfileSetupScreen() {
  const router = useRouter();
  const { user, updateUserProfile, error, setError } = useAuth();
  
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [address, setAddress] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validateInputs = (): boolean => {
    if (!name.trim()) {
      setError('Vui lòng nhập họ tên');
      return false;
    }
    if (!age.trim() || isNaN(Number(age))) {
      setError('Vui lòng nhập tuổi hợp lệ');
      return false;
    }
    if (!address.trim()) {
      setError('Vui lòng nhập địa chỉ');
      return false;
    }
    return true;
  };

  const handleSaveProfile = async () => {
    try {
      console.log('ProfileSetupScreen: Save profile button pressed');
      setError(null);
      
      if (!validateInputs()) {
        console.log('ProfileSetupScreen: Validation failed');
        return;
      }

      if (!user?.uid) {
        throw new Error('User ID not found');
      }

      setIsSubmitting(true);
      console.log('ProfileSetupScreen: Saving profile for user:', user.uid);
      
      await updateUserProfile({
        name,
        age: Number(age),
        address,
      });
      
      console.log('ProfileSetupScreen: Profile saved successfully, navigating to history');
      setIsSubmitting(false);
      
      // Navigate to history screen
      router.replace('/(tabs)' as any);
    } catch (err: any) {
      console.error('ProfileSetupScreen: Error saving profile:', err);
      setError(err.message || 'Failed to save profile');
      setIsSubmitting(false);
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
          <Text variant="displayMedium" style={styles.headerTitle}>
            Tạo hồ sơ
          </Text>
          <Text variant="bodyLarge" style={styles.subtitle}>
            Hoàn thành thông tin cá nhân
          </Text>
        </View>

        {/* Content */}
        <View style={styles.content}>
          {/* Name Input */}
          <Text variant="labelLarge" style={styles.label}>
            Họ tên
          </Text>
          <TextInput
            label="Nhập họ tên"
            value={name}
            onChangeText={(text) => {
              setName(text);
              if (error) setError(null);
            }}
            mode="outlined"
            style={styles.input}
            editable={!isSubmitting}
            outlineColor="#14B8A6"
            activeOutlineColor="#0D9488"
          />

          {/* Age Input */}
          <Text variant="labelLarge" style={styles.label}>
            Tuổi
          </Text>
          <TextInput
            label="Nhập tuổi"
            value={age}
            onChangeText={(text) => {
              setAge(text.replace(/[^0-9]/g, ''));
              if (error) setError(null);
            }}
            keyboardType="number-pad"
            mode="outlined"
            style={styles.input}
            editable={!isSubmitting}
            outlineColor="#14B8A6"
            activeOutlineColor="#0D9488"
          />

          {/* Address Input */}
          <Text variant="labelLarge" style={styles.label}>
            Địa chỉ
          </Text>
          <TextInput
            label="Nhập địa chỉ"
            value={address}
            onChangeText={(text) => {
              setAddress(text);
              if (error) setError(null);
            }}
            mode="outlined"
            style={styles.input}
            multiline
            numberOfLines={3}
            editable={!isSubmitting}
            outlineColor="#14B8A6"
            activeOutlineColor="#0D9488"
          />

          {/* Error Message */}
          {error && (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          {/* Save Button */}
          <Button
            mode="contained"
            onPress={handleSaveProfile}
            disabled={isSubmitting}
            loading={isSubmitting}
            style={styles.button}
            buttonColor="#14B8A6"
          >
            {isSubmitting ? 'Đang lưu...' : 'Lưu hồ sơ'}
          </Button>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  scrollContent: {
    flexGrow: 1,
    padding: 20,
  },
  header: {
    marginBottom: 30,
    marginTop: 20,
  },
  headerTitle: {
    color: '#151718',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    color: '#666',
  },
  content: {
    flex: 1,
  },
  label: {
    color: '#151718',
    marginTop: 16,
    marginBottom: 8,
    fontWeight: '600',
  },
  input: {
    backgroundColor: '#F5F5F5',
    marginBottom: 12,
  },
  errorBox: {
    backgroundColor: '#FEE2E2',
    borderRadius: 8,
    padding: 12,
    marginVertical: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#DC2626',
  },
  errorText: {
    color: '#DC2626',
    fontSize: 14,
  },
  button: {
    marginTop: 24,
    paddingVertical: 8,
  },
});
