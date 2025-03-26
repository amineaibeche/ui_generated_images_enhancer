// components/SettingsModal.js
import React, { useState, useEffect } from 'react';

const SettingsModal = ({ isOpen, onClose, onSave, initialModels }) => {
  const [selectedModels, setSelectedModels] = useState(initialModels);
  const [isAnimating, setIsAnimating] = useState(false); // State to track animation status

  // Handle model selection changes
  const handleModelChange = (category, model) => {
    setSelectedModels((prev) => ({
      ...prev,
      [category]: model,
    }));
  };

  // Trigger animations when the modal opens or closes
  useEffect(() => {
    if (isOpen) {
      setIsAnimating(true); // Start the animation
    } else {
      setTimeout(() => setIsAnimating(false), 300); // Wait for the animation to finish before hiding
    }
  }, [isOpen]);

  return (
    <>
      {/* Modal Overlay */}
      {isAnimating && (
        <div
          className={`fixed inset-0 z-50 flex items-center justify-center bg-black transition-opacity duration-300 ease-in-out ${
            isOpen ? 'bg-opacity-70' : 'bg-opacity-0 pointer-events-none'
          }`}
        >
          <div
            className={`bg-white dark:bg-gray-800 rounded-lg shadow-lg w-full max-w-md p-6 transform transition-transform duration-300 ease-in-out ${
              isOpen ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
            }`}
          >
            {/* Modal Header */}
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100 font-sans">
                Configure Models
              </h2>
              <button
                onClick={onClose}
                className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
              >
                <svg
                  className="w-6 h-6"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            {/* Modal Content */}
            <div className="space-y-4">
              {/* Most Models */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 font-sans">
                  Most Models
                </label>
                <div className="space-y-2">
                  {['ResNet-18', 'ResNet-50', 'Clip_AGIQA'].map((model) => (
                    <div key={model} className="flex items-center">
                      <input
                        type="radio"
                        id={`most-${model}`}
                        name="most-models"
                        value={model}
                        checked={selectedModels.most === model}
                        onChange={() => handleModelChange('most', model)}
                        className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                      />
                      <label
                        htmlFor={`most-${model}`}
                        className="ml-2 block text-sm text-gray-900 dark:text-gray-300 font-sans"
                      >
                        {model}
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Alignment Models */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 font-sans">
                  Alignment Models
                </label>
                <div className="space-y-2">
                  {['AlignNet-1_t5small_without_cross_attention', 'AlignNet-2_t5small_and_cross_attention', 'AlignNet-3_bert_and_cross_attention'].map((model) => (
                    <div key={model} className="flex items-center">
                      <input
                        type="radio"
                        id={`align-${model}`}
                        name="alignment-models"
                        value={model}
                        checked={selectedModels.alignment === model}
                        onChange={() => handleModelChange('alignment', model)}
                        className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                      />
                      <label
                        htmlFor={`align-${model}`}
                        className="ml-2 block text-sm text-gray-900 dark:text-gray-300 font-sans"
                      >
                        {model}
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Enhancement Models */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 font-sans">
                  Enhancement Models
                </label>
                <div className="space-y-2">
                  {['Prompt_enhancement', 'Error_map_based_on_alignement', 'Error_map_based_on_alignement_and_mos_quality'].map((model) => (
                    <div key={model} className="flex items-center">
                      <input
                        type="radio"
                        id={`enhance-${model}`}
                        name="enhancement-models"
                        value={model}
                        checked={selectedModels.enhancement === model}
                        onChange={() => handleModelChange('enhancement', model)}
                        className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                      />
                      <label
                        htmlFor={`enhance-${model}`}
                        className="ml-2 block text-sm text-gray-900 dark:text-gray-300 font-sans"
                      >
                        {model}
                      </label>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="mt-6 flex justify-end">
              <button
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-100 dark:hover:bg-gray-600 font-sans"
              >
                Cancel
              </button>
              <button
                onClick={() => onSave(selectedModels)}
                className="ml-2 px-4 py-2 text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 rounded-md font-sans"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default SettingsModal;