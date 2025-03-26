// components/Settings.js
import React from 'react';

const Settings = ({
  style,
  onStyleChange,
  quality,
  onQualityChange,
  detailLevel,
  onDetailLevelChange,
  guidanceScale,
  onGuidanceScaleChange,
  strength,
  onStrengthChange,
  steps, // Add steps as a prop
  onStepsChange, // Add a handler for steps changes
  onAction,
}) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      {/* Header */}
      <h3 className="text-lg font-semibold mb-4">Enhancement Settings</h3>

      {/* Content */}
      <div className="space-y-4">
        {/* Style */}
        <div>
          <label className="block text-sm font-medium mb-2">Style</label>
          <select
            value={style}
            onChange={onStyleChange}
            className="form-select w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3 text-gray-800 dark:text-gray-200"
          >
            <option value="artistic">Artistic</option>
            <option value="photorealistic">Photorealistic</option>
            <option value="minimalist">Minimalist</option>
          </select>
        </div>

        {/* Quality */}
        <div>
          <label className="block text-sm font-medium mb-2">Quality</label>
          <select
            value={quality}
            onChange={onQualityChange}
            className="form-select w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3 text-gray-800 dark:text-gray-200"
          >
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        {/* Detail Level */}
        <div>
          <label className="block text-sm font-medium mb-2">Detail Level</label>
          <input
            type="range"
            min="0"
            max="100"
            value={detailLevel}
            onChange={onDetailLevelChange}
            className="custom-slider w-full appearance-none h-2 bg-gradient-to-r from-blue-500 to-green-500 rounded-full outline-none"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>

        {/* Guidance Scale */}
        <div>
          <label className="block text-sm font-medium mb-2">Guidance Scale</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={guidanceScale}
            onChange={onGuidanceScaleChange}
            className="custom-slider w-full appearance-none h-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full outline-none"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>0</span>
            <span>1</span>
          </div>
        </div>

        {/* Strength */}
        <div>
          <label className="block text-sm font-medium mb-2">Strength</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={strength}
            onChange={onStrengthChange}
            className="custom-slider w-full appearance-none h-2 bg-gradient-to-r from-orange-500 to-red-500 rounded-full outline-none"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>0</span>
            <span>1</span>
          </div>
        </div>

        {/* Steps */}
        <div>
          <label className="block text-sm font-medium mb-2">Steps</label>
          <input
            type="range"
            min="0"
            max="50"
            value={steps}
            onChange={onStepsChange}
            className="custom-slider w-full appearance-none h-2 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-full outline-none"
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>0</span>
            <span>50</span>
          </div>
        </div>

        {/* Enhance Button */}
        <button
          onClick={onAction}
          className="w-full py-2 px-4 rounded-md text-white font-medium transition duration-300 bg-green-500 hover:bg-green-600"
        >
          Enhance
        </button>
      </div>
    </div>
  );
};

export default Settings;