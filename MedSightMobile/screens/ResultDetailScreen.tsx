import React, { useState } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  Image,
  Dimensions,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Text, Card, Button, List, Chip, Avatar, Divider, IconButton } from 'react-native-paper';
import { useNavigation, useRoute } from '@react-navigation/native';

const { width } = Dimensions.get('window');

export default function ResultDetailScreen() {
  const navigation = useNavigation<any>();
  const route = useRoute<any>();
  const { caseData } = route.params;
  
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const isCompleted = caseData.status === 'completed';

  const handleNextImage = () => {
    if (currentImageIndex < caseData.images.length - 1) {
      setCurrentImageIndex(currentImageIndex + 1);
    }
  };

  const handlePrevImage = () => {
    if (currentImageIndex > 0) {
      setCurrentImageIndex(currentImageIndex - 1);
    }
  };

  const handleGoBack = () => {
    navigation.goBack();
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Header with Back Button */}
        <View style={styles.header}>
          <TouchableOpacity onPress={handleGoBack}>
            <IconButton icon="arrow-left" size={28} iconColor="#14B8A6" />
          </TouchableOpacity>
          <Text variant="titleLarge" style={styles.headerTitle}>
            Chi tiết hồ sơ
          </Text>
          <View style={{ width: 40 }} />
        </View>

        {/* Image Section */}
        {caseData.images && caseData.images.length > 0 ? (
          <View style={styles.imageSection}>
            <View style={styles.imageContainer}>
              <Image
                source={{ uri: caseData.images[currentImageIndex] }}
                style={styles.image}
                resizeMode="cover"
              />
              {caseData.images.length > 1 && (
                <>
                  {/* Previous Button */}
                  {currentImageIndex > 0 && (
                    <TouchableOpacity
                      style={[styles.imageNavBtn, styles.prevBtn]}
                      onPress={handlePrevImage}
                    >
                      <Text style={styles.navBtnText}>‹</Text>
                    </TouchableOpacity>
                  )}

                  {/* Next Button */}
                  {currentImageIndex < caseData.images.length - 1 && (
                    <TouchableOpacity
                      style={[styles.imageNavBtn, styles.nextBtn]}
                      onPress={handleNextImage}
                    >
                      <Text style={styles.navBtnText}>›</Text>
                    </TouchableOpacity>
                  )}
                </>
              )}
            </View>

            {/* Image Counter */}
            {caseData.images.length > 1 && (
              <Text style={styles.imageCounter}>
                {currentImageIndex + 1} / {caseData.images.length}
              </Text>
            )}

            {/* Image Thumbnails */}
            {caseData.images.length > 1 && (
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                style={styles.thumbnailContainer}
              >
                {caseData.images.map((img: string, index: number) => (
                  <TouchableOpacity
                    key={index}
                    onPress={() => setCurrentImageIndex(index)}
                  >
                    <Image
                      source={{ uri: img }}
                      style={[
                        styles.thumbnail,
                        currentImageIndex === index && styles.thumbnailActive,
                      ]}
                    />
                  </TouchableOpacity>
                ))}
              </ScrollView>
            )}
          </View>
        ) : (
          <View style={[styles.imageContainer, styles.noImagePlaceholder]}>
            <Avatar.Icon
              icon="image-off-outline"
              size={64}
              style={{ backgroundColor: '#D1D5DB' }}
            />
            <Text style={styles.noImageText}>Không có hình ảnh</Text>
          </View>
        )}

        {/* Content Section */}
        <View style={styles.content}>
          {/* Case Info */}
          <Card style={styles.infoCard}>
            <Card.Title
              title="Thông tin hồ sơ"
              titleStyle={{ color: '#1F2937', fontWeight: 'bold' }}
              left={(props) => <List.Icon {...props} icon="information-outline" />}
            />
            <Card.Content>
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Mã hồ sơ:</Text>
                <Text style={styles.infoValue}>{caseData.id}</Text>
              </View>
              <Divider style={styles.divider} />
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Ngày khám:</Text>
                <Text style={styles.infoValue}>{caseData.created_at}</Text>
              </View>
              <Divider style={styles.divider} />
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Trạng thái:</Text>
                <Chip
                  icon={isCompleted ? 'check-circle' : 'clock-outline'}
                  style={{
                    backgroundColor: isCompleted ? '#D1FAE5' : '#FEF3C7',
                  }}
                  textStyle={{
                    color: isCompleted ? '#065F46' : '#78350F',
                    fontWeight: '600',
                  }}
                >
                  {isCompleted ? 'Đã hoàn thành' : 'Đang chờ'}
                </Chip>
              </View>
            </Card.Content>
          </Card>

          {/* Symptoms */}
          <Card style={styles.symptomsCard}>
            <Card.Title
              title="Triệu chứng ban đầu"
              titleStyle={{ color: '#1F2937', fontWeight: 'bold' }}
              left={(props) => <List.Icon {...props} icon="hospital-box" />}
            />
            <Card.Content>
              <Text variant="bodyMedium" style={styles.symptomsText}>
                {caseData.symptoms}
              </Text>
            </Card.Content>
          </Card>

          {/* AI Diagnosis */}
          {isCompleted && caseData.doctor_diagnosis && (
            <>
              <Card style={[styles.resultCard, styles.aiCard]}>
                <Card.Title
                  title="Gợi ý từ AI (MedSight)"
                  titleStyle={{ color: '#0369A1', fontWeight: 'bold' }}
                  left={(props) => <List.Icon {...props} icon="robot" color="#0369A1" />}
                />
                <Card.Content>
                  <Text variant="bodyLarge" style={styles.predictionText}>
                    <Text style={{ fontWeight: 'bold' }}>Dự đoán:</Text>{' '}
                    {caseData.doctor_diagnosis.ai_prediction}
                  </Text>
                  <Text variant="bodySmall" style={styles.confidenceText}>
                    (Mức độ tin cậy: Cao)
                  </Text>
                </Card.Content>
              </Card>

              {/* Doctor Diagnosis */}
              <Card style={[styles.resultCard, styles.doctorCard]}>
                <Card.Title
                  title="Kết luận bác sĩ"
                  titleStyle={{ color: '#059669', fontWeight: 'bold' }}
                  left={(props) => <List.Icon {...props} icon="doctor" color="#059669" />}
                  subtitle={`Dr. ${caseData.doctor_diagnosis.doctor_name}`}
                  subtitleStyle={{ fontSize: 12 }}
                />
                <Divider />
                <Card.Content>
                  <Text variant="labelLarge" style={styles.sectionLabel}>
                    Kết luận:
                  </Text>
                  <Text
                    variant="bodyMedium"
                    style={[styles.diagnosisText, { fontWeight: 'bold', color: '#059669' }]}
                  >
                    {caseData.doctor_diagnosis.conclusion}
                  </Text>

                  <Text variant="labelLarge" style={[styles.sectionLabel, { marginTop: 12 }]}>
                    Ghi chú:
                  </Text>
                  <Text variant="bodyMedium" style={styles.diagnosisText}>
                    {caseData.doctor_diagnosis.notes}
                  </Text>
                </Card.Content>
              </Card>

              {/* Recommendations */}
              <Card style={styles.recommendationCard}>
                <Card.Title
                  title="Hướng dẫn tiếp theo"
                  titleStyle={{ color: '#1F2937', fontWeight: 'bold' }}
                  left={(props) => (
                    <List.Icon {...props} icon="clipboard-check-outline" />
                  )}
                />
                <Card.Content>
                  <View style={styles.recommendationItem}>
                    <Text style={styles.recommendationBullet}>•</Text>
                    <Text style={styles.recommendationText}>
                      Tuân thủ theo hướng dẫn của bác sĩ
                    </Text>
                  </View>
                  <View style={styles.recommendationItem}>
                    <Text style={styles.recommendationBullet}>•</Text>
                    <Text style={styles.recommendationText}>
                      Liên hệ bác sĩ nếu có triệu chứng bất thường
                    </Text>
                  </View>
                  <View style={styles.recommendationItem}>
                    <Text style={styles.recommendationBullet}>•</Text>
                    <Text style={styles.recommendationText}>
                      Tái khám theo lịch hẹn
                    </Text>
                  </View>
                </Card.Content>
              </Card>
            </>
          )}

          {/* Pending State */}
          {!isCompleted && (
            <Card style={styles.pendingCard}>
              <Card.Content>
                <View style={styles.pendingContent}>
                  <Avatar.Icon
                    icon="hourglass-empty"
                    size={64}
                    style={{ backgroundColor: '#FEF3C7' }}
                  />
                  <Text
                    variant="headlineSmall"
                    style={[styles.pendingTitle, { marginTop: 16 }]}
                  >
                    Đang chờ bác sĩ
                  </Text>
                  <Text variant="bodyMedium" style={styles.pendingSubtitle}>
                    Bác sĩ sẽ sớm kiểm tra và phản hồi hồ sơ của bạn
                  </Text>
                </View>
              </Card.Content>
            </Card>
          )}

          {/* Action Buttons */}
          <View style={styles.actionButtons}>
            <Button
              mode="outlined"
              onPress={handleGoBack}
              style={styles.actionBtn}
              textColor="#14B8A6"
            >
              Quay lại
            </Button>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 8,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  headerTitle: {
    color: '#1F2937',
    fontWeight: '600',
    flex: 1,
    textAlign: 'center',
  },
  imageSection: {
    backgroundColor: '#FFFFFF',
    paddingVertical: 12,
  },
  imageContainer: {
    position: 'relative',
    height: 300,
    backgroundColor: '#000000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  noImagePlaceholder: {
    backgroundColor: '#F3F4F6',
  },
  noImageText: {
    color: '#9CA3AF',
    fontSize: 16,
    marginTop: 12,
  },
  imageNavBtn: {
    position: 'absolute',
    width: 44,
    height: 44,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  prevBtn: {
    left: 12,
  },
  nextBtn: {
    right: 12,
  },
  navBtnText: {
    color: '#FFFFFF',
    fontSize: 32,
    fontWeight: 'bold',
  },
  imageCounter: {
    textAlign: 'center',
    marginVertical: 8,
    color: '#6B7280',
    fontSize: 12,
    fontWeight: '600',
  },
  thumbnailContainer: {
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  thumbnail: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginRight: 8,
    borderWidth: 2,
    borderColor: '#E5E7EB',
  },
  thumbnailActive: {
    borderColor: '#14B8A6',
  },
  content: {
    padding: 12,
  },
  infoCard: {
    marginBottom: 12,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  infoLabel: {
    color: '#6B7280',
    fontWeight: '600',
    fontSize: 14,
  },
  infoValue: {
    color: '#1F2937',
    fontWeight: '600',
    fontSize: 14,
  },
  divider: {
    backgroundColor: '#E5E7EB',
    marginVertical: 4,
  },
  symptomsCard: {
    marginBottom: 12,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
  },
  symptomsText: {
    color: '#374151',
    lineHeight: 24,
  },
  resultCard: {
    marginBottom: 12,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    borderLeftWidth: 4,
  },
  aiCard: {
    borderLeftColor: '#0369A1',
  },
  doctorCard: {
    borderLeftColor: '#059669',
  },
  predictionText: {
    color: '#0369A1',
    fontWeight: 'bold',
    marginBottom: 4,
  },
  confidenceText: {
    color: '#6B7280',
    fontStyle: 'italic',
  },
  sectionLabel: {
    color: '#1F2937',
    fontWeight: '600',
    marginBottom: 4,
  },
  diagnosisText: {
    color: '#374151',
    lineHeight: 22,
  },
  recommendationCard: {
    marginBottom: 12,
    backgroundColor: '#F0FDF4',
    borderRadius: 12,
  },
  recommendationItem: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  recommendationBullet: {
    fontSize: 16,
    color: '#059669',
    marginRight: 12,
    fontWeight: 'bold',
  },
  recommendationText: {
    flex: 1,
    color: '#374151',
    fontSize: 14,
    lineHeight: 20,
  },
  pendingCard: {
    marginBottom: 12,
    backgroundColor: '#FFFBEB',
    borderRadius: 12,
  },
  pendingContent: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  pendingTitle: {
    color: '#78350F',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  pendingSubtitle: {
    color: '#92400E',
    textAlign: 'center',
    marginTop: 8,
  },
  actionButtons: {
    gap: 12,
    marginBottom: 20,
  },
  actionBtn: {
    borderWidth: 1.5,
    borderColor: '#14B8A6',
    paddingVertical: 4,
  },
});
