// pages/Signup.js
"use client";
import { useState } from 'react';
import firebase from '../firebase';
import { useRouter } from 'next/navigation';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
const Signup = () => {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSignup = async (e) => {
    e.preventDefault();
    try {
      await firebase.auth().createUserWithEmailAndPassword(email, password);
      router.push('/login')
    } catch (error) {
      toast(error.message);
    }
  };

  const handleLoginRedirect = () => {
    router.push('/login'); // Redirect to "/login"
  };

  return (
    <div className="flex justify-center items-center h-screen bg-sky-200">
      <ToastContainer />
      <form onSubmit={handleSignup} className="bg-black p-8 rounded shadow-md text-white">
        <h2 className="text-2xl mb-4">Sign Up</h2>
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
          Sign Up
        </button>
        <button
          type="button"
          className="text-white py-2 rounded"
          onClick={handleLoginRedirect}
        >
          Already have an account? Login
        </button>


      </form>
    </div>
  );
};

export default Signup;
