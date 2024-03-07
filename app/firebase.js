
import firebase from 'firebase/compat/app';
import 'firebase/compat/firestore';
import 'firebase/compat/auth';

// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAxP-JeYqudR-Qmeqph5ETuAbvVKPFeVQ4",
  authDomain: "text-to-image-and-video.firebaseapp.com",
  databaseURL: "https://text-to-image-and-video-default-rtdb.firebaseio.com",
  projectId: "text-to-image-and-video",
  storageBucket: "text-to-image-and-video.appspot.com",
  messagingSenderId: "101152243526",
  appId: "1:101152243526:web:b0649b1f11fb1ff5722f47",
  measurementId: "G-SXP821FLL9"
};


if (firebase.apps.length==0) {
  firebase.initializeApp(firebaseConfig);
}

export default firebase;
