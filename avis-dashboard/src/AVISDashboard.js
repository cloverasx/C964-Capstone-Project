/**
 * AVISDashboard.js
 * Main component for the Automated Vehicle Identification System
 * 
 * This component provides a user interface for uploading vehicle images
 * and displaying AI-powered predictions about the vehicle's make, model, and year.
 * It supports both file uploads (including drag-and-drop) and URL inputs.
 */

import React, { useState, useEffect } from 'react';
import { UploadCloud } from 'lucide-react';

// API endpoint for predictions
const API_ENDPOINT = '/api/predict';

// Recommendation system utility functions
const normalizeYear = (year) => {
  const currentYear = 2024;
  const oldestYear = 1990;
  return (year - oldestYear) / (currentYear - oldestYear);
};

const levenshteinDistance = (str1, str2) => {
  const matrix = Array(str2.length + 1).fill(null)
    .map(() => Array(str1.length + 1).fill(null));

  for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
  for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;

  for (let j = 1; j <= str2.length; j++) {
    for (let i = 1; i <= str1.length; i++) {
      const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
      matrix[j][i] = Math.min(
        matrix[j][i - 1] + 1,
        matrix[j - 1][i] + 1,
        matrix[j - 1][i - 1] + indicator
      );
    }
  }

  return matrix[str2.length][str1.length];
};

const stringSimilarity = (str1, str2) => {
  str1 = str1.toLowerCase();
  str2 = str2.toLowerCase();
  
  if (str1 === str2) return 1;
  
  const maxDist = Math.max(str1.length, str2.length);
  const dist = levenshteinDistance(str1, str2);
  
  return 1 - (dist / maxDist);
};

const calculateSimilarity = (vehicle1, vehicle2) => {
  const yearWeight = 0.3;
  const makeWeight = 0.3;
  const modelWeight = 0.4;

  let score = 0;

  // Year similarity
  const yearDiff = Math.abs(normalizeYear(vehicle1.year) - normalizeYear(vehicle2.year));
  score += (1 - yearDiff) * yearWeight;

  // Make similarity
  if (vehicle1.make.toLowerCase() === vehicle2.make.toLowerCase()) {
    score += makeWeight;
  }

  // Model similarity
  const modelSimilarity = stringSimilarity(vehicle1.model, vehicle2.model);
  score += modelSimilarity * modelWeight;

  return score;
};

const getVehicleDatabase = async () => {
  try {
    const response = await fetch('/api/vehicles');
    if (!response.ok) {
      throw new Error('Failed to fetch vehicle database');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching vehicle database:', error);
    return [];
  }
};

const getVehicleRecommendations = async (identifiedVehicle) => {
  if (!identifiedVehicle) return [];

  const vehicleDatabase = await getVehicleDatabase();
  if (!vehicleDatabase.length) return [];

  // Calculate similarity scores for all vehicles except the exact match
  const scoredVehicles = vehicleDatabase
    .filter(vehicle => 
      // Exclude exact match
      !(vehicle.make.toLowerCase() === identifiedVehicle.make.toLowerCase() &&
        vehicle.model.toLowerCase() === identifiedVehicle.model.toLowerCase() &&
        vehicle.year === identifiedVehicle.year)
    )
    .map(vehicle => ({
      ...vehicle,
      similarityScore: calculateSimilarity(identifiedVehicle, vehicle)
    }));

  // Sort by similarity score and take top 5
  return scoredVehicles
    .sort((a, b) => b.similarityScore - a.similarityScore)
    .slice(0, 5);
};

// VehicleRecommendations component
const VehicleRecommendations = ({ recommendations }) => {
  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h3 className="text-md font-semibold text-gray-800 mb-2">
        Similar Vehicles
      </h3>
      <div className="grid grid-cols-2 gap-2">
        {recommendations.map((vehicle, index) => (
          <div
            key={index}
            className="flex items-center justify-between p-2 bg-gray-50 rounded hover:bg-gray-100 transition-colors"
          >
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-gray-800 truncate">
                {vehicle.year} {vehicle.make} {vehicle.model}
              </div>
              <div className="bg-blue-100 text-blue-800 text-xs font-medium px-2 py-0.5 rounded-full inline-block mt-1">
                {(vehicle.similarityScore * 100).toFixed(0)}% Match
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Add this function near the top with other utility functions
const detectBodyStyle = (model) => {
  const modelLower = model.toLowerCase();
  
  // Common body style keywords
  if (modelLower.includes('convertible') || modelLower.includes('spyder') || modelLower.includes('roadster')) {
    return 'Convertible';
  }
  if (modelLower.includes('coupe')) {
    return 'Coupe';
  }
  if (modelLower.includes('sedan')) {
    return 'Sedan';
  }
  if (modelLower.includes('suv')) {
    return 'SUV';
  }
  if (modelLower.includes('hatchback')) {
    return 'Hatchback';
  }
  if (modelLower.includes('wagon')) {
    return 'Wagon';
  }
  if (modelLower.includes('van')) {
    return 'Van';
  }
  if (modelLower.includes('minivan')) {
    return 'Minivan';
  }
  if (modelLower.includes('crew cab')) {
    return 'Crew Cab';
  }
  if (modelLower.includes('extended cab')) {
    return 'Extended Cab';
  }
  if (modelLower.includes('regular cab')) {
    return 'Regular Cab';
  }
  
  // Default to Sedan if no body style is detected
  return 'Sedan';
};

// Main Dashboard Component
const AVISDashboard = () => {
  // State management
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  /**
   * Cleanup function for preview URLs to prevent memory leaks
   * @param {string} url - The URL to cleanup
   */
  const cleanupPreviewUrl = (url) => {
    if (url && url.startsWith('blob:')) {
      URL.revokeObjectURL(url);
    }
  };

  /**
   * Validates if a string is a valid image URL
   * @param {string} url - The URL to validate
   * @returns {boolean} - True if URL is valid
   */
  const isValidImageUrl = (url) => {
    try {
      const parsed = new URL(url);
      return parsed.protocol === 'http:' || parsed.protocol === 'https:';
    } catch {
      return false;
    }
  };

  /**
   * Handles the prediction request to the backend
   * @param {File|string} imageSource - The image file or URL to analyze
   */
  const getPredictions = async (imageSource) => {
    try {
      setLoading(true);
      setError('');
      setPredictions(null);

      const formData = new FormData();
      
      if (typeof imageSource === 'string' && !imageSource.startsWith('blob:')) {
        formData.append('image_url', imageSource);
      } else if (imageSource instanceof File) {
        formData.append('image', imageSource);
      } else {
        throw new Error('Invalid image source');
      }

      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
        },
        mode: 'cors',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP error! status: ${response.status}`);
      }

      const text = await response.text();
      const data = JSON.parse(text);

      if (!data || !data.make || !data.model || !data.year) {
        throw new Error('Invalid prediction data from server');
      }

      setPredictions(data);
      
      // Get vehicle recommendations based on predictions
      setLoadingRecommendations(true);
      try {
        const identifiedVehicle = {
          make: data.make.prediction,
          model: data.model.prediction,
          year: parseInt(data.year.prediction),
          body_style: detectBodyStyle(data.model.prediction)
        };
        
        const vehicleRecs = await getVehicleRecommendations(identifiedVehicle);
        setRecommendations(vehicleRecs);
      } catch (err) {
        console.error('Failed to get recommendations:', err);
        setRecommendations([]);
      } finally {
        setLoadingRecommendations(false);
      }
    } catch (err) {
      setError(err.message || 'Failed to analyze image');
      setPredictions(null);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handles file selection from the file input
   * @param {Event} event - The file input change event
   */
  const handleFileChange = async (event) => {
    cleanupPreviewUrl(previewUrl);
    
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setImageUrl('');
      setError('');
      
      const newPreviewUrl = URL.createObjectURL(file);
      setPreviewUrl(newPreviewUrl);
      
      await getPredictions(file);
    } else {
      setError('Please select a valid image file.');
    }
  };

  /**
   * Handles URL submission from the URL input form
   * @param {Event} event - The form submission event
   */
  const handleUrlSubmit = async (event) => {
    event.preventDefault();
    const url = imageUrl.trim();
    
    if (url && isValidImageUrl(url)) {
      cleanupPreviewUrl(previewUrl);
      setSelectedFile(null);
      setPreviewUrl(url);
      setError('');
      await getPredictions(url);
    } else {
      setError('Please enter a valid HTTP or HTTPS image URL');
    }
  };

  // Drag and drop event handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        setImageUrl('');
        setError('');
        
        const newPreviewUrl = URL.createObjectURL(file);
        setPreviewUrl(newPreviewUrl);
        
        await getPredictions(file);
      } else {
        setError('Please select a valid image file.');
      }
    }
  };

  /**
   * Renders a prediction card for make, model, or year
   * @param {string} title - The title of the prediction card
   * @param {Object} predictions - The predictions data
   * @returns {JSX.Element|null} - The rendered card or null
   */
  const renderPredictionCard = (title, predictions) => {
    if (!predictions) {
      return null;
    }
    
    const predictionData = predictions[title.toLowerCase()];
    if (!predictionData) {
      return null;
    }

    return (
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-md font-semibold mb-2">{title}</h3>
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="font-semibold">{predictionData.prediction}</span>
            <span className="text-sm text-gray-600">
              {(predictionData.confidence * 100).toFixed(1)}% confidence
            </span>
          </div>
          <div className="space-y-1">
            {predictionData.top_5.map(([label, conf], index) => (
              <div key={index} className="relative pt-0.5">
                <div className="flex justify-between mb-0.5">
                  <span className="text-sm font-medium">{label}</span>
                  <span className="text-sm font-medium">
                    {(conf * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="overflow-hidden h-1.5 text-xs flex rounded bg-blue-200">
                  <div
                    className="bg-blue-500"
                    style={{ width: `${conf * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // Cleanup preview URL on component unmount
  useEffect(() => {
    return () => {
      cleanupPreviewUrl(previewUrl);
    };
  }, [previewUrl]);

  // Component render
  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-6xl mx-auto space-y-4">
        {/* Header Section */}
        <div className="text-center mb-4">
          <h1 className="text-2xl font-bold mb-1">
            Automated Vehicle Identification System
          </h1>
          <p className="text-gray-600 text-sm">
            Upload an image or provide a URL to identify the vehicle
          </p>
        </div>

        {/* Input Section */}
        <div className="grid md:grid-cols-2 gap-4">
          {/* File Upload Area */}
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="text-lg font-semibold mb-2">Upload Image</h2>
            <div className="flex items-center justify-center w-full">
              <label 
                className={`flex flex-col w-full h-24 border-2 border-dashed 
                  ${isDragging 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'hover:bg-gray-50 hover:border-gray-300 border-gray-300'
                  } transition-colors duration-150 ease-in-out cursor-pointer`}
                onDragEnter={handleDragEnter}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="flex flex-col items-center justify-center pt-4">
                  <UploadCloud className={`w-6 h-6 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`} />
                  <p className={`pt-1 text-xs ${isDragging ? 'text-blue-500' : 'text-gray-400'}`}>
                    {isDragging ? 'Drop image here' : 'Drop image here or click to select'}
                  </p>
                </div>
                <input
                  type="file"
                  className="opacity-0"
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </label>
            </div>
          </div>

          {/* URL Input Area */}
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="text-lg font-semibold mb-2">Image URL</h2>
            <form onSubmit={handleUrlSubmit} className="space-y-2">
              <div>
                <input
                  type="url"
                  value={imageUrl}
                  onChange={(e) => setImageUrl(e.target.value)}
                  placeholder="Enter image URL"
                  className="w-full p-2 border rounded text-sm"
                />
              </div>
              <button
                type="submit"
                className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 text-sm"
                disabled={loading}
              >
                {loading ? 'Analyzing...' : 'Analyze'}
              </button>
            </form>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-3 text-sm">
            <div className="flex">
              <div className="flex-shrink-0">
                <UploadCloud className="h-4 w-4 text-red-400" />
              </div>
              <div className="ml-2">
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {(previewUrl || loading) && (
          <div className="grid md:grid-cols-5 gap-4">
            {/* Image Preview */}
            <div className="md:col-span-2 bg-white rounded-lg shadow p-4">
              <h2 className="text-lg font-semibold mb-2">Image Preview</h2>
              <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Vehicle"
                    className="w-full h-full object-cover"
                  />
                )}
              </div>
            </div>

            {/* Predictions and Recommendations Display */}
            <div className="md:col-span-3">
              {loading ? (
                <div className="bg-white rounded-lg shadow p-4">
                  <div className="flex items-center justify-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500" />
                    <span className="text-sm">Analyzing image...</span>
                  </div>
                </div>
              ) : (
                predictions && (
                  <div className="space-y-4">
                    {/* Predictions Grid */}
                    <div className="grid grid-cols-3 gap-2">
                      {renderPredictionCard('Make', predictions)}
                      {renderPredictionCard('Model', predictions)}
                      {renderPredictionCard('Year', predictions)}
                    </div>
                    
                    {/* Recommendations */}
                    {loadingRecommendations ? (
                      <div className="bg-white rounded-lg shadow p-4">
                        <div className="flex items-center justify-center space-x-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500" />
                          <span className="text-sm">Loading recommendations...</span>
                        </div>
                      </div>
                    ) : (
                      <VehicleRecommendations recommendations={recommendations} />
                    )}
                  </div>
                )
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AVISDashboard;