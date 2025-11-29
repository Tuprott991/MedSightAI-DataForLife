import React, { useState, useCallback } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Searchbar, Card, Text, Badge, Avatar, Divider, Chip } from 'react-native-paper';
import { useAuth } from '@/context/AuthContext';
import { useNavigation } from '@react-navigation/native';

// Mock data generator - will be replaced with API calls
const generateMockHistory = () => {
  const timestamp = Date.now();
  return [
    {
      id: `CASE_001_${timestamp}`,
      caseId: 'CASE_001',
      created_at: new Date().toISOString().split('T')[0],
      patient_name: 'Nguyễn Văn A',
      status: 'completed',
      symptoms: 'Đau ngực trái, khó thở khi vận động mạnh.',
      doctor_diagnosis: {
        doctor_name: 'ThS. BS Lê Thị B',
        conclusion: 'Viêm phổi thùy dưới phổi trái.',
        notes: 'Bệnh nhân cần nghỉ ngơi, uống thuốc theo đơn kê.',
        ai_prediction: 'Pneumonia (98.5%)',
      },
      images: [
        'https://prod-images-static.radiopaedia.org/images/1393683/2a3d60762419a4d339316d933390c2_jumbo.jpeg',
        'https://prod-images-static.radiopaedia.org/images/29424796/d8a43697e883833446820251347074_jumbo.jpeg',
      ],
    },
    {
      id: `CASE_002_${timestamp}`,
      caseId: 'CASE_002',
      created_at: new Date(Date.now() - 86400000).toISOString().split('T')[0],
      patient_name: 'Nguyễn Văn A',
      status: 'pending',
      symptoms: 'Ho khan kéo dài 2 tuần.',
      doctor_diagnosis: null,
      images: [],
    },
    {
      id: `CASE_003_${timestamp}`,
      caseId: 'CASE_003',
      created_at: new Date(Date.now() - 172800000).toISOString().split('T')[0],
      patient_name: 'Nguyễn Văn A',
      status: 'completed',
      symptoms: 'Sốt cao trên 39°C, đau đầu.',
      doctor_diagnosis: {
        doctor_name: 'TS. BS Trần Văn C',
        conclusion: 'Cảm cúm mùa A.',
        notes: 'Uống các thuốc hạ sốt, dùng kháng virus theo đơn.',
        ai_prediction: 'Influenza (92.3%)',
      },
      images: [],
    },
  ];
};

export default function HistoryScreen() {
  const navigation = useNavigation<any>();
  const { user, signOutUser } = useAuth();
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredData, setFilteredData] = useState(generateMockHistory());
  const [refreshing, setRefreshing] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  // Search handler
  const onChangeSearch = (query: string) => {
    setSearchQuery(query);
    const results = generateMockHistory().filter(
      (item) =>
        item.symptoms.toLowerCase().includes(query.toLowerCase()) ||
        item.id.toLowerCase().includes(query.toLowerCase()) ||
        item.patient_name.toLowerCase().includes(query.toLowerCase())
    );
    setFilteredData(results);
  };

  // Refresh handler - simulates API call
  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      // Simulate API call delay
      await new Promise((resolve) => setTimeout(resolve, 1500));
      
      // Generate new mock data (in real app, this would be an API call)
      const newData = generateMockHistory();
      setFilteredData(newData);
      
      // Show success message
      console.log('History refreshed successfully');
    } catch (error) {
      console.error('Error refreshing history:', error);
    } finally {
      setRefreshing(false);
    }
  }, []);

  // Load more handler
  const onLoadMore = useCallback(async () => {
    if (isLoadingMore) return;
    
    setIsLoadingMore(true);
    try {
      // Simulate API call delay
      await new Promise((resolve) => setTimeout(resolve, 1000));
      
      // Generate additional mock data
      const additionalData = generateMockHistory();
      setFilteredData((prevData) => [...prevData, ...additionalData]);
    } catch (error) {
      console.error('Error loading more history:', error);
    } finally {
      setIsLoadingMore(false);
    }
  }, [isLoadingMore]);

  // Handle item press - navigate to detail screen
  const handleItemPress = (item: any) => {
    navigation.navigate('ResultDetail', { caseData: item });
  };

  // Get status color
  const getStatusColor = (status: string) =>
    status === 'completed' ? '#10B981' : '#F59E0B';

  // Get status text
  const getStatusText = (status: string) =>
    status === 'completed' ? 'Đã có kết quả' : 'Đang chờ bác sĩ';

  // Render history item
  const renderItem = ({ item }: any) => (
    <TouchableOpacity
      onPress={() => handleItemPress(item)}
      activeOpacity={0.8}
    >
      <Card style={styles.card}>
        <Card.Title
          title={`Hồ sơ: ${item.id}`}
          subtitle={`Ngày: ${item.created_at}`}
          left={(props) => (
            <Avatar.Icon
              {...props}
              icon="file-document-outline"
              style={{ backgroundColor: '#14B8A6' }}
            />
          )}
          right={(props) => (
            <View style={{ marginRight: 16 }}>
              <Badge style={{ backgroundColor: getStatusColor(item.status) }}>
                {getStatusText(item.status)}
              </Badge>
            </View>
          )}
        />
        <Divider />
        <Card.Content style={{ paddingTop: 10 }}>
          <Text variant="bodyMedium" numberOfLines={2} style={{ color: '#555' }}>
            <Text style={{ fontWeight: 'bold' }}>Triệu chứng: </Text>
            {item.symptoms}
          </Text>
          {item.status === 'completed' && item.doctor_diagnosis && (
            <Text style={{ marginTop: 5, color: '#14B8A6', fontStyle: 'italic' }}>
              ✓ Bác sĩ {item.doctor_diagnosis.doctor_name} đã phản hồi
            </Text>
          )}
        </Card.Content>
      </Card>
    </TouchableOpacity>
  );

  // Render empty state
  const renderEmptyState = () => (
    <View style={styles.emptyContainer}>
      <Avatar.Icon
        icon="clipboard-outline"
        size={64}
        style={[styles.emptyIcon, { backgroundColor: '#14B8A6' }]}
      />
      <Text variant="headlineSmall" style={styles.emptyTitle}>
        Không tìm thấy hồ sơ
      </Text>
      <Text variant="bodyMedium" style={styles.emptySubtitle}>
        {searchQuery ? 'Thử điều chỉnh từ khóa tìm kiếm' : 'Bạn chưa có hồ sơ khám'}
      </Text>
    </View>
  );

  // Render footer (loading more indicator)
  const renderFooter = () => {
    if (!isLoadingMore) return null;
    return (
      <View style={styles.loadingFooter}>
        <Text style={styles.loadingText}>Đang tải thêm...</Text>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header Section */}
      <View style={styles.headerSection}>
        <View style={styles.headerTop}>
          <View>
            <Text variant="headlineMedium" style={styles.headerTitle}>
              Lịch sử khám
            </Text>
            <Text style={styles.headerSubtitle}>
              {user?.phoneNumber || 'Bệnh nhân'}
            </Text>
          </View>
          <TouchableOpacity
            style={styles.signOutBtn}
            onPress={() => signOutUser()}
          >
            <Text style={styles.signOutText}>Đăng xuất</Text>
          </TouchableOpacity>
        </View>

        {/* Search Bar */}
        <View style={styles.searchSection}>
          <Searchbar
            placeholder="Tìm theo triệu chứng, mã HS..."
            onChangeText={onChangeSearch}
            value={searchQuery}
            style={styles.searchBar}
            inputStyle={styles.searchInput}
            elevation={1}
            iconColor="#14B8A6"
          />
        </View>

        {/* Filter Chips */}
        {filteredData.length > 0 && (
          <View style={styles.filterSection}>
            <Chip
              selected={false}
              mode="outlined"
              style={styles.chip}
              textStyle={{ fontSize: 12 }}
            >
              Tổng: {filteredData.length}
            </Chip>
            <Chip
              selected={false}
              mode="outlined"
              style={styles.chip}
              textStyle={{ fontSize: 12 }}
            >
              Hoàn thành:{' '}
              {filteredData.filter((item) => item.status === 'completed').length}
            </Chip>
            <Chip
              selected={false}
              mode="outlined"
              style={styles.chip}
              textStyle={{ fontSize: 12 }}
            >
              Chờ:{' '}
              {filteredData.filter((item) => item.status === 'pending').length}
            </Chip>
          </View>
        )}
      </View>

      {/* History List */}
      {filteredData.length > 0 ? (
        <FlatList
          data={filteredData}
          renderItem={renderItem}
          keyExtractor={(item, index) => `${item.id}_${index}`}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={['#14B8A6']}
              tintColor="#14B8A6"
              title="Đang tải lại..."
              titleColor="#ECEDEE"
            />
          }
          onEndReached={onLoadMore}
          onEndReachedThreshold={0.5}
          ListFooterComponent={renderFooter}
          scrollIndicatorInsets={{ right: 1 }}
        />
      ) : (
        <ScrollView
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={['#14B8A6']}
              tintColor="#14B8A6"
              title="Đang tải lại..."
              titleColor="#ECEDEE"
            />
          }
          contentContainerStyle={styles.emptyScrollContent}
        >
          {renderEmptyState()}
        </ScrollView>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  headerSection: {
    backgroundColor: '#FFFFFF',
    paddingTop: 12,
    paddingHorizontal: 16,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  headerTitle: {
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#9CA3AF',
  },
  signOutBtn: {
    backgroundColor: '#FEE2E2',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  signOutText: {
    color: '#DC2626',
    fontSize: 12,
    fontWeight: '600',
  },
  searchSection: {
    marginBottom: 12,
  },
  searchBar: {
    backgroundColor: '#F3F4F6',
    borderRadius: 8,
    elevation: 0,
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  searchInput: {
    fontSize: 14,
    color: '#1F2937',
  },
  filterSection: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 8,
  },
  chip: {
    borderColor: '#D1D5DB',
    backgroundColor: '#F9FAFB',
  },
  listContent: {
    paddingHorizontal: 10,
    paddingVertical: 12,
  },
  card: {
    marginBottom: 12,
    borderRadius: 12,
    backgroundColor: '#FFFFFF',
    elevation: 2,
  },
  emptyScrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 40,
  },
  emptyIcon: {
    marginBottom: 16,
  },
  emptyTitle: {
    color: '#1F2937',
    fontWeight: '600',
    marginBottom: 8,
  },
  emptySubtitle: {
    color: '#9CA3AF',
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  loadingFooter: {
    paddingVertical: 20,
    alignItems: 'center',
  },
  loadingText: {
    color: '#9CA3AF',
    fontSize: 14,
  },
});
