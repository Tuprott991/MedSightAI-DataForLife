import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  Alert,
} from 'react-native';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'expo-router';

export default function DashboardScreen() {
  const { user, userProfile, signOutUser } = useAuth();
  const router = useRouter();

  const handleLogOut = async () => {
    try {
      console.log('Dashboard: Logout initiated');
      await signOutUser();
      console.log('Dashboard: Logout successful, navigating to login');
      router.replace('/login' as any);
    } catch (error: any) {
      console.error('Dashboard: Logout error:', error);
      Alert.alert('Lỗi', 'Không thể đăng xuất. Vui lòng thử lại.');
    }
  };

  const confirmLogout = () => {
    Alert.alert(
      'Xác nhận đăng xuất',
      'Bạn có chắc chắn muốn đăng xuất?',
      [
        {
          text: 'Hủy',
          style: 'cancel',
        },
        {
          text: 'Đăng xuất',
          onPress: handleLogOut,
          style: 'destructive',
        },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        {/* Welcome Section */}
        <View style={styles.welcomeSection}>
          <Text style={styles.welcomeTitle}>
            Chào mừng trở lại! 👋
          </Text>
          {userProfile?.name && (
            <Text style={styles.userName}>{userProfile.name}</Text>
          )}
          {user?.phoneNumber && (
            <Text style={styles.phoneNumber}>{user.phoneNumber}</Text>
          )}
        </View>

        {/* User Info Card */}
        {userProfile && (
          <View style={styles.infoCard}>
            <Text style={styles.infoTitle}>Thông tin cá nhân</Text>
            {userProfile.name && (
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Tên:</Text>
                <Text style={styles.infoValue}>{userProfile.name}</Text>
              </View>
            )}
            {userProfile.age && (
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Tuổi:</Text>
                <Text style={styles.infoValue}>{userProfile.age}</Text>
              </View>
            )}
            {userProfile.address && (
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Địa chỉ:</Text>
                <Text style={styles.infoValue}>{userProfile.address}</Text>
              </View>
            )}
          </View>
        )}

        {/* Spacer */}
        <View style={{ flex: 1 }} />

        {/* Logout Button */}
        <TouchableOpacity
          style={styles.logoutButton}
          onPress={confirmLogout}
          activeOpacity={0.7}
        >
          <Text style={styles.logoutButtonText}>Đăng xuất</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  content: {
    flex: 1,
    padding: 24,
  },
  welcomeSection: {
    marginTop: 40,
    marginBottom: 32,
  },
  welcomeTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: '#1F2937',
    marginBottom: 8,
  },
  userName: {
    fontSize: 24,
    fontWeight: '600',
    color: '#14B8A6',
    marginBottom: 4,
  },
  phoneNumber: {
    fontSize: 16,
    color: '#6B7280',
    marginTop: 4,
  },
  infoCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  infoTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 16,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  infoLabel: {
    fontSize: 16,
    color: '#6B7280',
    fontWeight: '500',
  },
  infoValue: {
    fontSize: 16,
    color: '#1F2937',
    fontWeight: '600',
  },
  logoutButton: {
    backgroundColor: '#EF4444',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 16,
    shadowColor: '#EF4444',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  logoutButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
  },
});
