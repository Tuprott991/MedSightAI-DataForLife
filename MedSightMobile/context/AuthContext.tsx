import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { 
  initializeAuth, 
  signInWithPhoneNumber, 
  signOut, 
  PhoneAuthProvider,
  signInWithCredential,
  onAuthStateChanged,
  RecaptchaVerifier,
  User,
} from 'firebase/auth';
import { initializeApp } from 'firebase/app';

// Firebase Config
const firebaseConfig = {
  apiKey: "AIzaSyCVzgm-z9EUtE1Nt5sD2I_Xr36zdL-6LmU",
  authDomain: "medai-1218f.firebaseapp.com",
  projectId: "medai-1218f",
  storageBucket: "medai-1218f.firebasestorage.app",
  messagingSenderId: "822629036608",
  appId: "1:822629036608:web:786657df5e803f92faa5a5",
  measurementId: "G-XP0ZRSEC94"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize auth without persistence for development
// This ensures users always start at login screen
const auth = initializeAuth(app, {
  persistence: [],  // Disable persistence - no stored sessions
});

export interface AuthUser {
  uid: string;
  phoneNumber: string | null;
  email: string | null;
  displayName: string | null;
}

interface AuthContextType {
  user: AuthUser | null;
  isLoading: boolean;
  isSignedIn: boolean;
  phoneNumber: string;
  setPhoneNumber: (phone: string) => void;
  verificationId: string | null;
  setVerificationId: (id: string | null) => void;
  sendOTP: (phoneNumber: string) => Promise<void>;
  verifyOTP: (otp: string) => Promise<void>;
  signOutUser: () => Promise<void>;
  error: string | null;
  setError: (error: string | null) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [phoneNumber, setPhoneNumber] = useState('');
  const [verificationId, setVerificationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initializeAuth = async () => {
      try {
        // Force sign out to clear any cached sessions on app startup
        await signOut(auth);
        console.log('Auth: Cleared cached sessions on startup');
      } catch (err) {
        console.log('Auth: No active session to clear');
      }
      
      const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
        console.log('Auth state changed:', currentUser ? `User: ${currentUser.uid}` : 'No user');
        if (currentUser) {
          setUser({
            uid: currentUser.uid,
            phoneNumber: currentUser.phoneNumber,
            email: currentUser.email,
            displayName: currentUser.displayName,
          });
        } else {
          setUser(null);
        }
        setIsLoading(false);
      });

      return unsubscribe;
    };

    let unsubscribe: any;
    initializeAuth().then((cleanup) => {
      unsubscribe = cleanup;
    });

    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, []);

  const sendOTP = async (phone: string) => {
    try {
      setError(null);
      setIsLoading(true);
      
      // Format phone number
      const formattedPhone = phone.startsWith('+') ? phone : `+${phone}`;
      
      // Create reCAPTCHA verifier (for web, this would be handled differently on mobile)
      const recaptchaVerifier = new RecaptchaVerifier(auth, 'recaptcha-container', {
        size: 'invisible',
      });

      const confirmationResult = await signInWithPhoneNumber(
        auth,
        formattedPhone,
        recaptchaVerifier
      );

      setVerificationId(confirmationResult.verificationId);
      setPhoneNumber(formattedPhone);
      setIsLoading(false);
    } catch (err: any) {
      setError(err.message || 'Failed to send OTP');
      setIsLoading(false);
      throw err;
    }
  };

  const verifyOTP = async (otp: string) => {
    try {
      setError(null);
      setIsLoading(true);

      if (!verificationId) {
        throw new Error('Verification ID not found');
      }

      const credential = PhoneAuthProvider.credential(verificationId, otp);
      await signInWithCredential(auth, credential);
      
      setIsLoading(false);
    } catch (err: any) {
      setError(err.message || 'Invalid OTP');
      setIsLoading(false);
      throw err;
    }
  };

  const signOutUser = async () => {
    try {
      setError(null);
      await signOut(auth);
      setUser(null);
      setPhoneNumber('');
      setVerificationId(null);
    } catch (err: any) {
      setError(err.message || 'Failed to sign out');
      throw err;
    }
  };

  const value: AuthContextType = {
    user,
    isLoading,
    isSignedIn: !!user,
    phoneNumber,
    setPhoneNumber,
    verificationId,
    setVerificationId,
    sendOTP,
    verifyOTP,
    signOutUser,
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
