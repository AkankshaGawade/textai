"use client"
import React, { useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import TextToImageGenerator from './TextToImage';

const Posts = () => {
  const [textInput, setTextInput] = useState('');
  const apiKey = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImJmNTliMWFkOTRhYTE3NTM2OGUzYTVhYzRmMDNiZTA1IiwiY3JlYXRlZF9hdCI6IjIwMjQtMDEtMzBUMDU6NTI6MTAuNTM2MjU1In0.J9yIc3XALrbR73yfK0tL34PgXh3M5rUXCXXHiWZSmZk'; // Replace with your actual API key

  const handleTextToImageGeneration = async () => {
    try {
      console.log('Generating image with text:', textInput);
      // Call the TextToImageGenerator component with apiKey and textInput
      <TextToImageGenerator apiKey={apiKey} text={textInput} />;
    } catch (error) {
      console.error('Error generating image:', error);
      toast.error('Error generating image');
    }
  };

  return (
    <div className="container">
      <TextToImageGenerator apiKey={apiKey} text={textInput} />
      <ToastContainer />
    </div>
  );
};

export default Posts;
