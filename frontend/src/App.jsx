import { useState } from 'react'
import Header from './components/Header'
import PredictionForm from './components/PredictionForm'
import ResultCard from './components/ResultCard'

function App() {
  const [predictionResult, setPredictionResult] = useState(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <Header />
      
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Classify Iris Flowers with AI
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Enter the measurements of an Iris flower and our machine learning models will predict its species: 
            <span className="font-semibold text-green-600"> Setosa</span>,
            <span className="font-semibold text-purple-600"> Versicolor</span>, or
            <span className="font-semibold text-pink-600"> Virginica</span>.
          </p>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Prediction Form */}
          <div>
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <span className="text-2xl">📊</span>
              Input Features
            </h3>
            <PredictionForm onPrediction={setPredictionResult} />
          </div>

          {/* Result Card */}
          <div>
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Prediction Results
            </h3>
            {predictionResult ? (
              <ResultCard result={predictionResult} />
            ) : (
              <div className="bg-white rounded-2xl shadow-xl p-12 border border-gray-100 text-center">
                <div className="text-6xl mb-4">🔮</div>
                <p className="text-gray-500 text-lg font-medium">
                  Fill in the form and click "Predict Species" to see results
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Info Cards */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <div className="text-3xl mb-3">🌸</div>
            <h4 className="font-bold text-gray-800 mb-2">Setosa</h4>
            <p className="text-sm text-gray-600">
              Known for its smaller petals and distinctive features. Easily distinguishable from other species.
            </p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <div className="text-3xl mb-3">🌺</div>
            <h4 className="font-bold text-gray-800 mb-2">Versicolor</h4>
            <p className="text-sm text-gray-600">
              Medium-sized flowers with moderate measurements across all features.
            </p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <div className="text-3xl mb-3">🌷</div>
            <h4 className="font-bold text-gray-800 mb-2">Virginica</h4>
            <p className="text-sm text-gray-600">
              Largest of the three species, characterized by longer petals and sepals.
            </p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-16 bg-white border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600">
            <p className="text-sm">
              Built with Django, React, and Machine Learning | Powered by Random Forest & SVC Models
            </p>
            <p className="text-xs text-gray-500 mt-2">
              © 2025 Iris ML Deployment. Professional AI Solutions.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
