// Dashboard.js
import React, { useState } from 'react';
import Sidebar from '../partials/Sidebar';
import Header from '../partials/Header';
import Datepicker from '../components/Datepicker';
import Banner from '../partials/Banner';
import OriginalImage from '../components/OriginalImage';
import Settings from '../components/Settings'; // Import the updated Settings component
import Output from '../components/Output';
import SettingsModal from '../components/SettingsModal'; // Import the SettingsModal

function Dashboard() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeOption, setActiveOption] = useState('enhance'); // Tracks the active mode
  const [selectedImage, setSelectedImage] = useState(null);
  const [enhancedImage, setEnhancedImage] = useState(null);
  const [enhancedImageUrl, setEnhancedImageUrl] = useState(null);
  const [style, setStyle] = useState('artistic');
  const [quality, setQuality] = useState('high');
  const [detailLevel, setDetailLevel] = useState(50);
  const [guidanceScale, setGuidanceScale] = useState(0.85); // Default guidance scale
  const [strength, setStrength] = useState(0.5);
  const [steps, setSteps] = useState(25); // Default strength
  const [prompt, setPrompt] = useState('');
  const [isImageUploaded, setIsImageUploaded] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModels, setSelectedModels] = useState({
    most: 'ResNet-18',
    alignment: 'AlignNet-1_t5small_without_cross_attention',
    enhancement: 'Prompt_enhancement',
  });

  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false); // Controls modal visibility

  // Handle image upload
  const handleImageChange = (e) => {
    if (e.target.files[0]) {
      setSelectedImage(URL.createObjectURL(e.target.files[0]));
      setIsImageUploaded(true);
    }
  };


  const handleAction = async () => {
    setIsLoading(true);
    setEnhancedImageUrl(null);
  
    try {
      const formData = new FormData();
      const imageFile = document.querySelector('input[type="file"]')?.files[0];
  
      // Append all parameters to the FormData object
      formData.append('image', imageFile);
      formData.append('prompt', prompt);
      formData.append('steps', steps);
      formData.append('quality', quality);
      formData.append('detail_level', detailLevel);
      formData.append('guidance_scale', guidanceScale);
      formData.append('strength', strength);
      formData.append('perceptual_model', selectedModels.most);
      formData.append('alignment_model', selectedModels.alignment);
      formData.append('enhancement_model', selectedModels.enhancement);
  
      // Determine the API endpoint based on the active option
      const endpoint =
        activeOption === 'enhance'
          ? 'http://localhost:8000/api/enhance/'
          : 'http://localhost:8000/api/evaluate/';
  
      // Send the POST request to the backend
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'API request failed');
      }
  
      const result = await response.json();
  
      // Update the state with the enhanced image URL
      if (activeOption === 'enhance') {
        // Add a cache-busting query parameter
        const cacheBuster = Date.now(); // Use current timestamp
        const enhancedImageUrlWithCacheBuster = `${result.enhanced_image_url}?v=${cacheBuster}`;
        setEnhancedImageUrl(enhancedImageUrlWithCacheBuster); // Store the updated URL
      } else {
        setEvaluationResults(result);
      }
    } catch (error) {
      console.error('Error:', error.message);
      alert(`An error occurred: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      {/* <Sidebar sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} /> */}

      {/* Content area */}
      <div className="relative flex flex-col flex-1 overflow-y-auto overflow-x-hidden">
        {/* Header */}
        <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />

        <main className="grow">
          <div className="px-4 sm:px-6 lg:px-8 py-8 w-full max-w-9xl mx-auto">
            {/* Top Controls */}
            <div className="sm:flex sm:justify-between sm:items-center mb-8">
              <div className="mb-4 sm:mb-0">
                <h1 className="text-2xl md:text-3xl text-gray-800 dark:text-gray-100 font-bold">
                  Perceptual Image Enhancer
                </h1>
              </div>

              <div className="flex gap-2 items-center">
                {/* Other controls */}
                <Datepicker align="right" />

                {/* Settings Icon Button */}
                <button
                  onClick={() => setIsModalOpen(true)} // Open the modal
                  className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 rounded-md transition duration-300 ease-in-out"
                  title="Settings"
                >
                  {/* Gear Icon */}
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
                      d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                  </svg>
                </button>

                {/* Toggle Switch with Animation */}
                <div
                  className="relative w-24 h-12 bg-gray-200 dark:bg-gray-700 rounded-full cursor-pointer"
                  onClick={() =>
                    setActiveOption((prev) =>
                      prev === 'enhance' ? 'evaluate' : 'enhance'
                    )
                  }
                >
                  {/* Sliding Container */}
                  <div
                    className={`
                      absolute inset-y-0 left-0 w-1/2 flex items-center transition-all duration-500 ease-in-out transform
                      rounded-full bg-white shadow-md
                      ${activeOption === 'enhance'
                        ? 'translate-x-0 border-2 border-green-500'
                        : 'translate-x-[2.75rem] border-2 border-blue-500'
                      }
                    `}
                  >
                    {/* Icon with Rotation Animation */}
                    <svg
                      className="w-6 h-6 mx-auto transition-transform duration-500 ease-in-out"
                      style={{
                        color:
                          activeOption === 'enhance' ? '#10B981' : '#3B82F6',
                        transform:
                          activeOption === 'enhance'
                            ? 'rotate(0deg)'
                            : 'rotate(360deg)',
                      }}
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      {activeOption === 'enhance' ? (
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M16 11V7a4 4 0 00-8 0v4m8 0h-4m4 0l-5 5m5-5l5 5"
                        />
                      ) : (
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      )}
                    </svg>
                  </div>
                </div>
              </div>
            </div>

            {/* Conditional Layout with Transition */}
            <div className="transition-all duration-500 ease-in-out">
              {activeOption === 'enhance' ? (
                // Enhancement Mode Layout
                <div
                  key="enhance"
                  className="grid grid-cols-1 md:grid-cols-[36%_28%_36%] gap-6 mb-8 transition-all duration-500 ease-in-out"
                >
                  {/* Original Image */}
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold mb-4">Original Image</h3>
                    <OriginalImage
                      prompt={prompt}
                      onPromptChange={setPrompt}
                      selectedImage={selectedImage}
                      onImageChange={handleImageChange}
                    />
                  </div>

                  {/* Settings Component */}
                  <Settings
                    style={style}
                    onStyleChange={(e) => setStyle(e.target.value)}
                    quality={quality}
                    onQualityChange={(e) => setQuality(e.target.value)}
                    detailLevel={detailLevel}
                    onDetailLevelChange={(e) =>
                      setDetailLevel(parseInt(e.target.value))
                    }
                    guidanceScale={guidanceScale}
                    onGuidanceScaleChange={(e) =>
                      setGuidanceScale(parseFloat(e.target.value))
                    }
                    strength={strength}
                    onStrengthChange={(e) =>
                      setStrength(parseFloat(e.target.value))
                    }
                    steps={steps} // Pass steps as a prop
                    onStepsChange={(e) => setSteps(parseInt(e.target.value))} // Handle steps changes
                    onAction={handleAction}
                  />

                  {/* Output Component */}
                  {/* Output Component */}
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 w-full">
  <h3 className="text-lg font-semibold mb-4 w-full">Enhanced Image</h3>
  {isLoading && activeOption === 'enhance' && (
    <div
      className="relative w-full aspect-square bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden"
      style={{
        background: `linear-gradient(90deg, rgba(16,185,129,0.2) 0%, rgba(59,130,246,0.2) 50%, rgba(16,185,129,0.2) 100%)`,
        backgroundSize: '200% auto',
        animation: 'gradient-animation 2s linear infinite',
      }}
    >
      <div className="absolute inset-0 flex justify-center items-center w-full h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-blue-500"></div>
      </div>
    </div>
  )}
  {!isLoading && enhancedImageUrl && (
    <>
      <img
        src={enhancedImageUrl}
        alt="Enhanced"
        className="w-full h-auto object-contain rounded-lg shadow-md"
      />
      <div className="mt-4 w-full">
        <div className="text-center text-lg font-semibold text-gray-700 dark:text-gray-300 mb-4 w-full">
          Download
        </div>
        <div className="flex justify-around gap-4 w-full">
          {/* 1K Button */}
          <button
            className="w-1/3 py-2 text-white font-medium rounded-lg transition duration-300 ease-in-out"
            style={{
              background: `linear-gradient(90deg, #10B981 0%,rgb(13, 117, 120) 100%)`,
              backgroundSize: '200% 100%',
              transition: 'background-position 0.5s',
            }}
            onMouseEnter={(e) => e.target.style.backgroundPosition = '100% 0%'}
            onMouseLeave={(e) => e.target.style.backgroundPosition = '0% 0%'}
            onClick={() => handleDownload('1k')}
          >
            1
          </button>
          {/* 2K Button */}
          <button
            className="w-1/3 py-2 text-white font-medium rounded-lg transition duration-300 ease-in-out"
            style={{
              background: `linear-gradient(90deg, rgb(13, 117, 120)  0%,rgb(72, 138, 245) 100%)`,
              backgroundSize: '200% 100%',
              transition: 'background-position 0.5s',
            }}
            onMouseEnter={(e) => e.target.style.backgroundPosition = '100% 0%'}
            onMouseLeave={(e) => e.target.style.backgroundPosition = '0% 0%'}
            onClick={() => handleDownload('2k')}
          >
            2
          </button>
          {/* 4K Button */}
          <button
            className="w-1/3 py-2 text-white font-medium rounded-lg transition duration-300 ease-in-out"
            style={{
              background: `linear-gradient(90deg, rgb(72, 138, 245) 0%,rgb(11, 59, 136) 100%)`,
              backgroundSize: '200% 100%',
              transition: 'background-position 0.5s',
            }}
            onMouseEnter={(e) => e.target.style.backgroundPosition = '100% 0%'}
            onMouseLeave={(e) => e.target.style.backgroundPosition = '0% 0%'}
            onClick={() => handleDownload('4k')}
          >
            4
          </button>
        </div>
      </div>
    </>
  )}
  {!isLoading && !enhancedImageUrl && (
    <div className="py-4 text-gray-500 w-full text-center">
      No results yet
    </div>
  )}
</div>

{/* CSS Animation for Gradient Background */}
<style>
  {`
    @keyframes gradient-animation {
      0% {
        background-position: 0% 50%;
      }
      100% {
        background-position: 100% 50%;
      }
    }
  `}
</style>
                </div>
              ) : (
                // Evaluation Mode Layout
                <div
                  key="evaluate"
                  className="grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-6 mb-8 transition-all duration-500 ease-in-out"
                >
                  {/* Left Side: Original Image and Run Evaluation Button */}
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold mb-4">Original Image</h3>
                    <OriginalImage
                      prompt={prompt}
                      onPromptChange={setPrompt}
                      selectedImage={selectedImage}
                      onImageChange={handleImageChange}
                    />

                    {/* Run Evaluation Button */}
                    {isImageUploaded && (
                      <button
                        className="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition duration-300 ease-in-out"
                        onClick={handleAction}
                      >
                        Run Evaluation
                      </button>
                    )}
                  </div>

                  {/* Right Side: Evaluation Results */}
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold mb-4">Evaluation Results</h3>
                    {isLoading && (
                      <div className="flex justify-center items-center py-4">
                        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
                      </div>
                    )}
                    {!isLoading && evaluationResults && (
                      <div className="space-y-4">
                        <div className="flex justify-between">
                          <span className="text-sm font-medium text-gray-500">
                            Perceptual Quality:
                          </span>
                          <span className="text-sm font-medium text-gray-800 dark:text-gray-100">
                            {parseFloat(evaluationResults.perceptual_quality).toFixed(2)}/5
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm font-medium text-gray-500">
                            Alignment Quality:
                          </span>
                          <span className="text-sm font-medium text-gray-800 dark:text-gray-100">
                            {parseFloat(evaluationResults.alignment_quality).toFixed(2)}/5
                          </span>
                        </div>
                        {/* Add other metrics here */}
                      </div>
                    )}
                    {!isLoading && !evaluationResults && (
                      <div className="text-center py-4">No results yet</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>

        {/* Settings Modal */}
        <SettingsModal
          isOpen={isModalOpen} // Controls the visibility of the modal
          onClose={() => setIsModalOpen(false)} // Close the modal without saving
          onSave={(selectedModels) => {
            console.log('Selected Models:', selectedModels); // Log the selected models
            setSelectedModels(selectedModels); // Update the state with the selected models
            setIsModalOpen(false); // Close the modal after saving
          }}
          initialModels={{
            most: 'ResNet-18', // Default perceptual model
            alignment: 'AlignNet-1_t5small_without_cross_attention', // Default alignment model
            enhancement: 'Prompt_enhancement', // Default enhancement model
          }}
        />

        <Banner />
      </div>
    </div>
  );
}

export default Dashboard;