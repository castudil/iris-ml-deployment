import { useState } from 'react';
import { API_ENDPOINTS } from '../config/api';

export default function PredictionForm({ onPrediction }) {
  const [formData, setFormData] = useState({
    sepalLength: '',
    sepalWidth: '',
    petalLength: '',
    petalWidth: '',
    modelName: 'rf_model'
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const features = [
        parseFloat(formData.sepalLength),
        parseFloat(formData.sepalWidth),
        parseFloat(formData.petalLength),
        parseFloat(formData.petalWidth)
      ];

      // Validate all features are valid numbers
      if (features.some(f => isNaN(f))) {
        throw new Error('Please fill in all feature values with valid numbers');
      }

      const response = await fetch(API_ENDPOINTS.predict, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: formData.modelName,
          features: features
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const result = await response.json();
      onPrediction(result);
    } catch (err) {
      setError(err.message);
      console.error('Prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      sepalLength: '',
      sepalWidth: '',
      petalLength: '',
      petalWidth: '',
      modelName: 'rf_model'
    });
    setError(null);
    onPrediction(null);
  };

  const loadSampleData = (type) => {
    const samples = {
      setosa: { sepalLength: '5.1', sepalWidth: '3.5', petalLength: '1.4', petalWidth: '0.2' },
      versicolor: { sepalLength: '6.4', sepalWidth: '3.2', petalLength: '4.5', petalWidth: '1.5' },
      virginica: { sepalLength: '7.2', sepalWidth: '3.6', petalLength: '6.1', petalWidth: '2.5' }
    };
    setFormData(prev => ({ ...prev, ...samples[type] }));
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Model Selection */}
        <div>
          <label htmlFor="modelName" className="block text-sm font-semibold text-gray-700 mb-2">
            Select Model
          </label>
          <select
            id="modelName"
            name="modelName"
            value={formData.modelName}
            onChange={handleChange}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors bg-white"
          >
            <option value="rf_model">Random Forest</option>
            <option value="svc_model">Support Vector Classifier</option>
          </select>
        </div>

        {/* Feature Inputs Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="sepalLength" className="block text-sm font-semibold text-gray-700 mb-2">
              Sepal Length (cm)
            </label>
            <input
              type="number"
              id="sepalLength"
              name="sepalLength"
              step="0.1"
              min="0"
              value={formData.sepalLength}
              onChange={handleChange}
              placeholder="e.g., 5.1"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              required
            />
          </div>

          <div>
            <label htmlFor="sepalWidth" className="block text-sm font-semibold text-gray-700 mb-2">
              Sepal Width (cm)
            </label>
            <input
              type="number"
              id="sepalWidth"
              name="sepalWidth"
              step="0.1"
              min="0"
              value={formData.sepalWidth}
              onChange={handleChange}
              placeholder="e.g., 3.5"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              required
            />
          </div>

          <div>
            <label htmlFor="petalLength" className="block text-sm font-semibold text-gray-700 mb-2">
              Petal Length (cm)
            </label>
            <input
              type="number"
              id="petalLength"
              name="petalLength"
              step="0.1"
              min="0"
              value={formData.petalLength}
              onChange={handleChange}
              placeholder="e.g., 1.4"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              required
            />
          </div>

          <div>
            <label htmlFor="petalWidth" className="block text-sm font-semibold text-gray-700 mb-2">
              Petal Width (cm)
            </label>
            <input
              type="number"
              id="petalWidth"
              name="petalWidth"
              step="0.1"
              min="0"
              value={formData.petalWidth}
              onChange={handleChange}
              placeholder="e.g., 0.2"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              required
            />
          </div>
        </div>

        {/* Sample Data Buttons */}
        <div className="border-t border-gray-200 pt-6">
          <p className="text-sm font-medium text-gray-700 mb-3">Quick Fill Sample Data:</p>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => loadSampleData('setosa')}
              className="px-4 py-2 bg-green-50 text-green-700 rounded-lg hover:bg-green-100 transition-colors text-sm font-medium"
            >
              Setosa Sample
            </button>
            <button
              type="button"
              onClick={() => loadSampleData('versicolor')}
              className="px-4 py-2 bg-purple-50 text-purple-700 rounded-lg hover:bg-purple-100 transition-colors text-sm font-medium"
            >
              Versicolor Sample
            </button>
            <button
              type="button"
              onClick={() => loadSampleData('virginica')}
              className="px-4 py-2 bg-pink-50 text-pink-700 rounded-lg hover:bg-pink-100 transition-colors text-sm font-medium"
            >
              Virginica Sample
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg">
            <p className="text-sm font-medium">⚠️ {error}</p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-4 pt-4">
          <button
            type="submit"
            disabled={isLoading}
            className="flex-1 bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-4 rounded-lg font-semibold hover:from-blue-700 hover:to-blue-800 focus:ring-4 focus:ring-blue-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Predicting...
              </span>
            ) : (
              '🔮 Predict Species'
            )}
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="px-6 py-4 border-2 border-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-50 transition-colors"
          >
            Reset
          </button>
        </div>
      </form>
    </div>
  );
}
