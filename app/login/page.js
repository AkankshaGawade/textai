// pages/Login.js
"use client";
import { useState } from 'react';
import firebase from '../firebase';
import { useRouter } from 'next/navigation';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const Login = () => {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      await firebase.auth().signInWithEmailAndPassword(email, password);
      toast(" User Login  Successful ");
      router.push('/trynow');
    } catch (error) {
      // Handle login error
      toast(error.message);
    }
  };

  const handleSignRedirect = () => {
    router.push('/signup'); 
  };
  return (
    <div className="flex justify-center items-center h-screen bg-sky-200">
      <ToastContainer />
      <form onSubmit={handleLogin} className="bg-black p-8 rounded shadow-md text-white">
        <h2 className="text-2xl mb-4 ">Login</h2>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full mb-4 p-2 rounded border text-black"
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full mb-4 p-2 rounded border text-black"
        />
        <button type="submit" className="w-full bg-blue-300 text-black py-2 rounded">
          Log In
        </button>

        <button
          type="button"
          className="text-white py-2 rounded"
          onClick={handleSignRedirect}
        >
            Signup here 
        </button>
        

      </form>
    </div>
  );
};

export default Login;
