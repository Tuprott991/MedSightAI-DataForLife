import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import 'react-native-reanimated';
import React from 'react';
import { ActivityIndicator, View } from 'react-native';
import { AuthProvider, useAuth } from '@/context/AuthContext';
import { useColorScheme } from '@/hooks/use-color-scheme';

export const unstable_settings = {
  anchor: '(tabs)',
};

function RootLayoutNav() {
  const colorScheme = useColorScheme();
  const { isSignedIn, isLoading, user } = useAuth();

  console.log('RootLayoutNav: isLoading =', isLoading, ', isSignedIn =', isSignedIn, ', user =', user);

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#151718' }}>
        <ActivityIndicator size="large" color="#14B8A6" />
      </View>
    );
  }

  return (
    <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
      <Stack 
        screenOptions={{ headerShown: false }}
        initialRouteName={isSignedIn ? '(tabs)' : 'login'}
      >
        {/* Auth Screens */}
        <Stack.Screen name="login" />
        <Stack.Screen name="signup" options={{ presentation: 'card' }} />
        
        {/* App Screens */}
        <Stack.Screen name="(tabs)" />
        <Stack.Screen
          name="ResultDetail"
          options={{ presentation: 'card' }}
        />
      </Stack>
      <StatusBar style="auto" />
    </ThemeProvider>
  );
}

export default function RootLayout() {
  return (
    <AuthProvider>
      <RootLayoutNav />
    </AuthProvider>
  );
}
