import React, { useState, useEffect, useRef } from 'react';
import { Sun, Moon, Eye, EyeOff } from 'lucide-react';

const SensoryWorld = () => {
  const [isAwake, setIsAwake] = useState(false);
  const [eyesOpen, setEyesOpen] = useState(false);
  const [brightness, setBrightness] = useState(0);
  const [visualInput, setVisualInput] = useState([]);
  const [emotionalResponse, setEmotionalResponse] = useState('');
  const worldRef = useRef(null);

  // Simulate gradual eye opening
  const openEyes = () => {
    if (!isAwake) return;
    setEyesOpen(true);
    let bright = 0;
    const interval = setInterval(() => {
      bright += 0.1;
      setBrightness(bright);
      if (bright >= 1) clearInterval(interval);
    }, 500);
  };

  // Generate shapes in the visual field
  const generateShapes = () => {
    const shapes = [];
    for (let i = 0; i < 5; i++) {
      shapes.push({
        type: Math.random() > 0.5 ? 'circle' : 'square',
        x: Math.random() * 80 + 10,
        y: Math.random() * 80 + 10,
        size: Math.random() * 20 + 10,
        color: `hsl(${Math.random() * 360}, 70%, 50%)`
      });
    }
    setVisualInput(shapes);
  };

  // Emotional response to visual input
  useEffect(() => {
    if (eyesOpen && visualInput.length > 0) {
      const responses = ['!', 'o', '~'];
      const intensity = Math.floor(Math.random() * 3) + 1;
      const response = responses[Math.floor(Math.random() * responses.length)].repeat(intensity);
      setEmotionalResponse(response);
    }
  }, [visualInput, eyesOpen]);

  return (
    <div className="w-full max-w-2xl mx-auto p-6 bg-gray-100 rounded-lg">
      <div className="flex justify-between items-center mb-4">
        <button
          onClick={() => setIsAwake(!isAwake)}
          className="flex items-center gap-2 px-4 py-2 rounded bg-blue-500 text-white"
        >
          {isAwake ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          {isAwake ? 'Awake' : 'Sleeping'}
        </button>
        
        <button
          onClick={openEyes}
          disabled={!isAwake}
          className={`flex items-center gap-2 px-4 py-2 rounded ${
            isAwake ? 'bg-green-500 text-white' : 'bg-gray-300 text-gray-600'
          }`}
        >
          {eyesOpen ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
          {eyesOpen ? 'Eyes Open' : 'Eyes Closed'}
        </button>
      </div>

      <div 
        ref={worldRef}
        className="relative w-full h-64 bg-white rounded-lg overflow-hidden"
        style={{
          filter: `brightness(${brightness})`,
          transition: 'filter 0.5s ease'
        }}
      >
        {eyesOpen && visualInput.map((shape, i) => (
          <div
            key={i}
            className={`absolute ${shape.type === 'circle' ? 'rounded-full' : ''}`}
            style={{
              left: `${shape.x}%`,
              top: `${shape.y}%`,
              width: `${shape.size}px`,
              height: `${shape.size}px`,
              backgroundColor: shape.color,
              transition: 'all 0.5s ease'
            }}
          />
        ))}
      </div>

      <div className="mt-4 text-center text-2xl font-mono">
        {emotionalResponse}
      </div>

      {eyesOpen && (
        <button
          onClick={generateShapes}
          className="mt-4 w-full px-4 py-2 bg-purple-500 text-white rounded"
        >
          Show New Shapes
        </button>
      )}
    </div>
  );
};

export default SensoryWorld;