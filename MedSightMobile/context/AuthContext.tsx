import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { auth, db, app } from '@/services/firebase';
import { signOut, onAuthStateChanged, PhoneAuthProvider, signInWithCredential } from 'firebase/auth';
import { doc, getDoc, setDoc, collection } from 'firebase/firestore';

// Mock storage for users (simulates Firestore for offline/testing)
const mockUserDatabase: Record<string, any> = {};

// Test phone numbers and OTP configuration
const TEST_PHONES = ['+16505551234', '0999999999'];
const TEST_OTP = '123456';

export interface AuthUser {
  uid: string;
  phoneNumber: string | null;
  email: string | null;
  displayName: string | null;
}

export interface UserProfile {
  uid: string;
  phoneNumber: string;
  name?: string;
  age?: number;
  address?: string;
  createdAt: string;
  updatedAt?: string;
}

interface AuthContextType {
  user: AuthUser | null;
  userProfile: UserProfile | null;
  isLoading: boolean;
  isSignedIn: boolean;
  isNewUser: boolean;
  phoneNumber: string;
  setPhoneNumber: (phone: string) => void;
  verificationId: string | null;
  setVerificationId: (id: string | null) => void;
  sendOTP: (phoneNumber: string, appVerifier: any) => Promise<string>;
  verifyOTP: (otp: string) => Promise<any>;
  signOutUser: () => Promise<void>;
  updateUserProfile: (profile: Partial<UserProfile>) => Promise<void>;
  error: string | null;
  setError: (error: string | null) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isNewUser, setIsNewUser] = useState(false);
  const [phoneNumber, setPhoneNumber] = useState('');
  const [verificationId, setVerificationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Check if user exists in Firestore (with mock fallback)
  const checkUserExists = async (uid: string): Promise<boolean> => {
    try {
      console.log('Auth: Checking if user exists, UID:', uid);
      
      try {
        // Try to get from Firestore
        const userDocRef = doc(db, 'users', uid);
        const userDoc = await getDoc(userDocRef);
        
        if (userDoc.exists()) {
          console.log('Auth: User found in Firestore');
          setUserProfile(userDoc.data() as UserProfile);
          setIsNewUser(false);
          return true;
        }
      } catch (firestoreErr) {
        console.log('Auth: Firestore unavailable, checking mock database');
      }
      
      // Fallback to mock database
      if (mockUserDatabase[uid]) {
        console.log('Auth: User found in mock database');
        setUserProfile(mockUserDatabase[uid]);
        setIsNewUser(false);
        return true;
      }
      
      console.log('Auth: User NOT found - new user');
      setIsNewUser(true);
      return false;
    } catch (err) {
      console.error('Auth: Error checking user:', err);
      return false;
    }
  };

  useEffect(() => {
    const initializeAuthListener = async () => {
      // Listen to auth state changes
      const unsubscribe = onAuthStateChanged(auth, async (currentUser: any) => {
        console.log('Auth state changed:', currentUser ? `User: ${currentUser.uid}` : 'No user');
        
        if (currentUser) {
          const authUser: AuthUser = {
            uid: currentUser.uid,
            phoneNumber: currentUser.phoneNumber,
            email: currentUser.email,
            displayName: currentUser.displayName,
          };
          setUser(authUser);
          
          // Check if user exists in Firestore
          await checkUserExists(currentUser.uid);
        } else {
          setUser(null);
          setUserProfile(null);
          setIsNewUser(false);
        }
        setIsLoading(false);
      });

      return unsubscribe;
    };

    let unsubscribe: any;
    initializeAuthListener().then((cleanup) => {
      unsubscribe = cleanup;
      console.log('Auth: onAuthStateChanged listener attached');
    });

    return () => {
      console.log('Auth: Cleaning up auth listener');
      if (unsubscribe) unsubscribe();
    };
  }, []);

  const sendOTP = async (phone: string, appVerifier: any): Promise<string> => {
    try {
      console.log('Auth: sendOTP called with phone:', phone);
      
      setError(null);
      setIsLoading(true);
      
      // Format phone number: remove leading 0, add +84
      let formattedPhone = phone.trim();
      if (formattedPhone.startsWith('0')) {
        formattedPhone = '+84' + formattedPhone.slice(1);
      } else if (!formattedPhone.startsWith('+')) {
        formattedPhone = '+' + formattedPhone;
      }
      
      console.log('Auth: Formatted phone:', formattedPhone);
      setPhoneNumber(formattedPhone);
      
      // If ReCAPTCHA not available, use mock OTP for development
      if (!appVerifier) {
        console.warn('⚠️  ReCAPTCHA not available - using mock OTP verification for development');
        console.log('Auth: Test phone numbers: +84 945 876 079, +1 650-555-1234');
        console.log('Auth: Test OTP: 123456');
        
        const mockVerificationId = 'mock-verification-' + Date.now();
        setVerificationId(mockVerificationId);
        setIsLoading(false);
        
        return mockVerificationId;
      }
      
      // Call Firebase to send OTP via SMS
      const phoneProvider = new PhoneAuthProvider(auth);
      console.log('Auth: Sending OTP with Firebase PhoneAuthProvider...');
      
      const verificationId = await phoneProvider.verifyPhoneNumber(
        formattedPhone,
        appVerifier
      );
      
      console.log('Auth: OTP sent successfully! Verification ID:', verificationId);
      setVerificationId(verificationId);
      setIsLoading(false);
      
      return verificationId;
      
    } catch (err: any) {
      console.error('Auth: sendOTP error:', err);
      console.error('Auth: Error code:', err.code);
      console.error('Auth: Error message:', err.message);
      
      // Provide user-friendly error messages
      let errorMessage = err.message || 'Failed to send OTP';
      if (err.code === 'auth/invalid-phone-number') {
        errorMessage = 'Invalid phone number format';
      } else if (err.code === 'auth/too-many-requests') {
        errorMessage = 'Too many requests. Please try again later.';
      } else if (err.code === 'auth/operation-not-supported-in-this-environment') {
        errorMessage = 'Phone authentication not supported in this environment';
      }
      
      setError(errorMessage);
      setIsLoading(false);
      throw err;
    }
  };

  const verifyOTP = async (otp: string): Promise<any> => {
    try {
      console.log('Auth: verifyOTP called with OTP length:', otp.length);
      setError(null);
      setIsLoading(true);

      if (!verificationId) {
        throw new Error('Verification ID not found. Please send OTP first.');
      }

      // Check if this is a mock verification ID (development mode)
      if (verificationId.startsWith('mock-')) {
        console.warn('⚠️  Using mock OTP verification');
        
        // Accept test OTP
        if (otp !== TEST_OTP && otp !== '123456') {
          setError('Invalid OTP. Test OTP: 123456');
          setIsLoading(false);
          throw new Error('Invalid OTP');
        }
        
        // Create mock user
        const mockUid = 'mock-user-' + Date.now();
        const authUser: AuthUser = {
          uid: mockUid,
          phoneNumber: phoneNumber,
          email: null,
          displayName: null,
        };
        
        setUser(authUser);
        mockUserDatabase[mockUid] = {
          uid: mockUid,
          phoneNumber: phoneNumber,
        };
        
        await checkUserExists(mockUid);
        setIsLoading(false);
        
        return { uid: mockUid };
      }

      // Real Firebase verification
      const credential = PhoneAuthProvider.credential(verificationId, otp);
      const userCredential = await signInWithCredential(auth, credential);
      console.log('Auth: User signed in successfully:', userCredential.user.uid);
      
      // Create auth user object
      const authUser: AuthUser = {
        uid: userCredential.user.uid,
        phoneNumber: userCredential.user.phoneNumber,
        email: userCredential.user.email,
        displayName: userCredential.user.displayName,
      };
      
      setUser(authUser);
      
      // Check if user exists in Firestore
      await checkUserExists(userCredential.user.uid);
      
      setIsLoading(false);
      return userCredential.user;
    } catch (err: any) {
      console.error('Auth: verifyOTP error:', err);
      console.error('Auth: Error code:', err.code);
      
      // Provide user-friendly error messages
      let errorMessage = err.message || 'Invalid OTP';
      if (err.code === 'auth/invalid-verification-code') {
        errorMessage = 'Invalid verification code. Please try again.';
      } else if (err.code === 'auth/code-expired') {
        errorMessage = 'Verification code has expired. Please request a new one.';
      }
      
      setError(errorMessage);
      setIsLoading(false);
      throw err;
    }
  };

  const updateUserProfile = async (profile: Partial<UserProfile>) => {
    try {
      if (!user) {
        throw new Error('No user logged in');
      }

      console.log('Auth: updateUserProfile called for UID:', user.uid);
      
      const updatedProfile: UserProfile = {
        uid: user.uid,
        phoneNumber: phoneNumber,
        ...profile,
        createdAt: userProfile?.createdAt || new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };

      try {
        // Try to save to Firestore
        const userDocRef = doc(db, 'users', user.uid);
        await setDoc(userDocRef, updatedProfile);
        console.log('Auth: User profile saved to Firestore');
      } catch (firestoreErr) {
        console.log('Auth: Firestore unavailable, saving to mock database');
        mockUserDatabase[user.uid] = updatedProfile;
      }
      
      setUserProfile(updatedProfile);
      setIsNewUser(false);
    } catch (err: any) {
      console.error('Auth: Error updating user profile:', err);
      setError(err.message || 'Failed to update profile');
      throw err;
    }
  };

  const signOutUser = async () => {
    try {
      console.log('Auth: signOutUser called');
      setError(null);
      await signOut(auth);
      console.log('Auth: Firebase signOut successful');
      setUser(null);
      setUserProfile(null);
      setPhoneNumber('');
      setVerificationId(null);
      setIsNewUser(false);
      console.log('Auth: State cleared - user should be null now');
    } catch (err: any) {
      console.error('Auth: signOut error:', err);
      setError(err.message || 'Failed to sign out');
      throw err;
    }
  };

  const value: AuthContextType = {
    user,
    userProfile,
    isLoading,
    isSignedIn: !!user,
    isNewUser,
    phoneNumber,
    setPhoneNumber,
    verificationId,
    setVerificationId,
    sendOTP,
    verifyOTP,
    signOutUser,
    updateUserProfile,
    error,
    setError,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
