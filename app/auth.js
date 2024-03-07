// app/auth.js
"use client"
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
// app/auth.js

// app/auth.js

import { useAuthState } from 'firebase/auth';
import { firebase } from '../app/firebase'; 

const AuthComponent = ({ children }) => {
  const router = useRouter();
  const auth = firebase.auth(); // Get the auth instance from Firebase
  const [user, loading, error] = useAuthState(auth);

  useEffect(() => {
    if (!loading && !user) {
      router.replace('/login');
    }
  }, [loading, user, router]);

  return <>{children}</>;
};

export default AuthComponent;
