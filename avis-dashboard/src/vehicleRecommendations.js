/**
 * Vehicle recommendation system that calculates similarity between vehicles
 * based on make, model, and year.
 */

// Normalize the year to a 0-1 scale for comparison
const normalizeYear = (year) => {
  const currentYear = 2024;
  const oldestYear = 1990;
  return (year - oldestYear) / (currentYear - oldestYear);
};

// Levenshtein distance calculation for string similarity
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

// Calculate string similarity using Levenshtein distance
const stringSimilarity = (str1, str2) => {
  str1 = str1.toLowerCase();
  str2 = str2.toLowerCase();
  
  if (str1 === str2) return 1;
  
  const maxDist = Math.max(str1.length, str2.length);
  const dist = levenshteinDistance(str1, str2);
  
  return 1 - (dist / maxDist);
};

// Calculate similarity score between two vehicles
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

// Get vehicle database from API
const getVehicleDatabase = async () => {
  try {
    const response = await fetch('http://127.0.0.1:8000/api/vehicles');
    if (!response.ok) {
      throw new Error('Failed to fetch vehicle database');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching vehicle database:', error);
    return [];
  }
};

// Get vehicle recommendations
export const getVehicleRecommendations = async (identifiedVehicle) => {
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