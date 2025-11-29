// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
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
const analytics = getAnalytics(app);