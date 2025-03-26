import React from 'react';

const Output = ({ activeOption, enhancedImage }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">
        {activeOption === 'enhance' ? 'Enhanced' : 'Evaluation'} Result
      </h3>
      <div className="flex items-center justify-center">
        {activeOption === 'enhance' ? (
          enhancedImage ? (
            <img
              src={enhancedImage}
              alt="Enhanced"
              className="w-full h-auto max-h-[500px] object-contain rounded-lg"
            />
          ) : (
            <div className="flex items-center justify-center text-gray-400 h-64">
              Result will appear here
            </div>
          )
        ) : (
          <div className="flex items-center justify-center text-gray-400 h-64">
            Evaluation metrics will appear here
          </div>
        )}
      </div>
    </div>
  );
};

export default Output;