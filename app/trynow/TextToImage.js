import React, { useState } from 'react';
import MonsterApiClient from 'monsterapi';

const TextToImageGenerator = ({ apiKey }) => {
  const [textInput, setTextInput] = useState('');
  const [generatedImage, setGeneratedImage] = useState(null);
  const [gifUrl, setGifUrl] = useState('');
  const [loadingImage, setLoadingImage] = useState(false);
  const [loadingVideo, setLoadingVideo] = useState(false);

  const handleTextToImageGeneration = async () => {
    try {
      if (!textInput.trim()) {
        alert('Please enter text for image generation.');
        return;
      }
      setLoadingImage(true);
      const client = new MonsterApiClient(apiKey);
      const input = {
        prompt: textInput,
        samples: 1,
      };
      const response = await client.generate('txt2img', input);
      if (response && response.output && response.output.length > 0) {
        setGeneratedImage(response.output[0]);
      }
    } catch (error) {
      console.error('Error generating image:', error);
      alert('Error generating image. Please try again.');
    } finally {
      setLoadingImage(false);
    }
  };

  const handleTextToVideoGeneration = async () => {
    try {
      if (!textInput.trim()) {
        alert('Please enter text for video generation.');
        return;
      }
      setLoadingVideo(true);
      const response = await fetch('http://127.0.0.1:5000/generate_gif', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt: textInput }) 
      });
      if (!response.ok) {
        throw new Error('Failed to generate GIF');
      }
      const data = await response.json();
      setGifUrl(data.gif_url);
    } catch (error) {
      console.error(error);
    } finally {
      setLoadingVideo(false);
    }
  };

  const resetInput = () => {
    setTextInput('');
    setGeneratedImage(null);
    setGifUrl('');
  };


  return (
    <div className="text-to-image-generator">
      <textarea
        className="text-input text-black"
        placeholder="Please enter text for image or video generation"
        value={textInput}
        onChange={(e) => setTextInput(e.target.value)}
      />
      <div className="button-container">
        <button className="generate-btn" onClick={handleTextToImageGeneration}>
          Generate Image
        </button>
        <button className="generate-btn" onClick={handleTextToVideoGeneration}>
          Generate Video
        </button>
        <button className="reset-btn" onClick={resetInput}>
          Reset
        </button>
      </div>
      {loadingImage && <div className="loader"><div className="spinner"></div>Generating Image...</div>}
      {loadingVideo && <div className="loader"><div className="spinner"></div>Generating Video...</div>}
      {generatedImage && (
        <div className="generated-content">
          <h2 className='text-white'>Generated Image</h2>
          <img src={generatedImage} alt="Generated" />
        </div>
      )}
      {gifUrl && (
        <div className="generated-content">
          <h2 className='text-white'>Generated Video</h2>
          <img src={gifUrl} alt="Generated GIF" />
        </div>
      )}
      <style jsx>{`
        .text-to-image-generator {
          max-width: 600px;
          margin: 20px auto;
          padding: 20px;
          border: 1px solid #ccc;
          border-radius: 8px;
        }
        .text-input {
          width: 100%;
          padding: 10px;
          margin-bottom: 10px;
          border: 1px solid #ccc;
          border-radius: 5px;
          resize: vertical;
        }
        .button-container {
          display: flex;
          margin-bottom: 10px;
        }
        .generate-btn {
          flex: 1;
          margin-right: 10px;
          background-color: #007bff;
          color: #fff;
          border: none;
          border-radius: 4px;
          padding: 10px 15px;
          cursor: pointer;
          transition: background-color 0.3s ease;
        }
        .generate-btn:last-child {
          margin-right: 0;
        }
        .generate-btn:hover {
          background-color: #0056b3;
        }
        .loader {
          margin-bottom: 10px;
          text-align: center;
          font-style: italic;
          color: #777;
        }
        .loader .spinner {
          border: 5px solid #f3f3f3; /* Light grey */
          border-top: 5px solid #3498db; /* Blue */
          border-radius: 50%;
          width: 30px;
          height: 30px;
          animation: spin 1s linear infinite;
          margin-right: 10px;
          display: inline-block;
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        .generated-content {
          text-align: center;
          margin-bottom: 20px;
        }
        .generated-content h2 {
          font-size: 24px;
          color: #333;
        }
        .generated-content img {
          max-width: 100%;
          height: auto;
        }
      `}</style>
    </div>
  );
};

export default TextToImageGenerator;
