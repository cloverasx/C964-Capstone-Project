import React, { useState } from 'react';
import { Popover } from '@headlessui/react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Pie, Line } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  ChartTooltip,
  Legend
);

// Tooltip wrapper component for consistent styling
const TooltipWrapper = ({ children, content }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div 
      className="relative w-full h-full"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <Popover className="relative w-full h-full">
        <Popover.Button as="div" className="w-full h-full outline-none">
          {children}
        </Popover.Button>
        {isOpen && (
          <Popover.Panel static className="absolute z-10 px-3 py-2 text-sm bg-gray-900 text-white rounded shadow-lg -top-2 left-1/2 transform -translate-x-1/2 -translate-y-full">
            {content}
            <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-gray-900 rotate-45" />
          </Popover.Panel>
        )}
      </Popover>
    </div>
  );
};

const VehicleVisualizations = ({ predictions, recommendations }) => {
  // Prepare data for make distribution chart
  const makeDistributionData = () => {
    if (!recommendations || recommendations.length === 0) return null;
    
    const makeCounts = recommendations.reduce((acc, vehicle) => {
      acc[vehicle.make] = (acc[vehicle.make] || 0) + 1;
      return acc;
    }, {});

    return {
      labels: Object.keys(makeCounts),
      datasets: [
        {
          data: Object.values(makeCounts),
          backgroundColor: [
            'rgba(255, 99, 132, 0.5)',
            'rgba(54, 162, 235, 0.5)',
            'rgba(255, 206, 86, 0.5)',
            'rgba(75, 192, 192, 0.5)',
            'rgba(153, 102, 255, 0.5)',
          ],
          borderColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)',
          ],
          borderWidth: 1,
        },
      ],
    };
  };

  // Prepare data for year confidence timeline
  const yearTimelineData = () => {
    if (!predictions || !predictions.year || !predictions.year.top_5) return null;

    // Get all unique years from the dataset (you may want to adjust this range based on your data)
    const allYears = Array.from({ length: 35 }, (_, i) => 1990 + i);
    
    // Create a map of year to confidence, defaulting to 0 for years not in top 5
    const yearConfidenceMap = new Map(allYears.map(year => [year, 0]));
    
    // Update confidences from top 5 predictions
    predictions.year.top_5.forEach(([year, confidence]) => {
      yearConfidenceMap.set(parseInt(year), confidence);
    });

    // Convert to sorted array of year-confidence pairs
    const yearPredictions = Array.from(yearConfidenceMap.entries())
      .map(([year, confidence]) => ({
        year,
        confidence
      }))
      .sort((a, b) => a.year - b.year);

    return {
      labels: yearPredictions.map(v => v.year.toString()),
      datasets: [
        {
          label: 'Year Confidence',
          data: yearPredictions.map(v => ({
            x: v.year,
            y: v.confidence,
            label: `Year ${v.year}`
          })),
          backgroundColor: 'rgba(54, 162, 235, 0.5)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1,
          pointRadius: 4,
          fill: true,
          tension: 0.4, // Add some curve to the line
        },
      ],
    };
  };

  const makeOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Distribution of Vehicle Makes in Recommendations',
      },
    },
  };

  const timelineOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Year Prediction Confidence Distribution',
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const point = context.raw;
            const value = (point.y * 100).toFixed(1);
            return `${point.label} (${value}%)`;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        title: {
          display: true,
          text: 'Confidence Score',
        },
        ticks: {
          callback: value => `${(value * 100).toFixed(0)}%`
        }
      },
      x: {
        title: {
          display: true,
          text: 'Year',
        },
      },
    },
  };

  const makeData = makeDistributionData();
  const timelineData = yearTimelineData();

  if (!makeData || !timelineData) {
    return null;
  }

  return (
    <div className="grid md:grid-cols-2 gap-4">
      <TooltipWrapper content="Shows the distribution of manufacturers among similar vehicles">
        <div className="bg-white p-3 rounded-lg shadow">
          <div className="h-[160px]">
            <Pie data={makeData} options={makeOptions} />
          </div>
        </div>
      </TooltipWrapper>
      
      <TooltipWrapper content="Shows the confidence distribution across all possible years for the identified vehicle">
        <div className="bg-white p-3 rounded-lg shadow">
          <div className="h-[160px]">
            <Line data={timelineData} options={timelineOptions} />
          </div>
        </div>
      </TooltipWrapper>
    </div>
  );
};

export default VehicleVisualizations; 