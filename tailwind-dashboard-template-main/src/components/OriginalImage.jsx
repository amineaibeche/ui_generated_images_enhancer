import React from 'react';

const OriginalImage = ({ prompt, onPromptChange, selectedImage, onImageChange }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 space-y-4">
      {/* Prompt Section */}
      <div>
        <h3 className="text-lg font-semibold mb-2">Prompt</h3>
        <input
          type="text"
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          className="block w-full p-2 border rounded-lg text-sm dark:bg-gray-700 dark:border-gray-600"
          placeholder="Describe your image requirements..."
        />
      </div>

      {/* Image Upload Section */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Upload Original Image</h3>
        <input
          type="file"
          accept="image/*"
          onChange={onImageChange}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gray-100 dark:file:bg-gray-700 dark:file:text-gray-400"
        />
      </div>

      {/* Image Preview Section */}
      {selectedImage && (
        <div className="mt-4 flex justify-center items-center">
          <div className="text-center">
            <h3 className="text-lg font-semibold mb-2">Image Preview</h3>
            <img
              src={selectedImage}
              alt="Preview"
              className="max-w-full h-auto rounded-lg mx-auto"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default OriginalImage;