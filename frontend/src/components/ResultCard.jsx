export default function ResultCard({ result }) {
  if (!result) return null;

  const getSpeciesColor = (species) => {
    const colors = {
      setosa: 'from-green-500 to-emerald-600',
      versicolor: 'from-purple-500 to-violet-600',
      virginica: 'from-pink-500 to-rose-600'
    };
    return colors[species] || 'from-gray-500 to-gray-600';
  };

  const getSpeciesEmoji = (species) => {
    const emojis = {
      setosa: '🌸',
      versicolor: '🌺',
      virginica: '🌷'
    };
    return emojis[species] || '🌼';
  };

  const maxProba = Math.max(...(result.prediction_proba || [0]));
  const confidence = (maxProba * 100).toFixed(1);

  return (
    <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100 animate-fadeIn">
      <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
        <span className="text-3xl">✨</span>
        Prediction Result
      </h3>

      {/* Main Prediction Card */}
      <div className={`bg-gradient-to-br ${getSpeciesColor(result.prediction)} text-white rounded-xl p-8 mb-6 shadow-lg`}>
        <div className="text-center">
          <p className="text-sm font-medium opacity-90 mb-2">Predicted Species</p>
          <div className="flex items-center justify-center gap-3 mb-4">
            <span className="text-5xl">{getSpeciesEmoji(result.prediction)}</span>
            <h2 className="text-4xl font-bold capitalize">{result.prediction}</h2>
          </div>
          <div className="bg-white/20 backdrop-blur-sm rounded-lg px-6 py-3 inline-block">
            <p className="text-sm font-medium">Confidence: <span className="text-2xl font-bold">{confidence}%</span></p>
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 mb-4">
        <p className="text-sm text-gray-600">
          <span className="font-semibold">Model Used:</span>{' '}
          <span className="text-blue-700 font-medium">
            {result.model_name === 'rf_model' ? 'Random Forest' : 'Support Vector Classifier'}
          </span>
        </p>
      </div>

      {/* Probability Distribution */}
      {result.prediction_proba && result.prediction_proba.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Probability Distribution:</h4>
          {['setosa', 'versicolor', 'virginica'].map((species, idx) => {
            const proba = result.prediction_proba[idx] || 0;
            const percentage = (proba * 100).toFixed(1);
            return (
              <div key={species} className="space-y-1">
                <div className="flex justify-between items-center text-sm">
                  <span className="font-medium text-gray-700 capitalize flex items-center gap-2">
                    <span>{getSpeciesEmoji(species)}</span>
                    {species}
                  </span>
                  <span className="font-semibold text-gray-900">{percentage}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                  <div
                    className={`h-2.5 rounded-full transition-all duration-500 bg-gradient-to-r ${getSpeciesColor(species)}`}
                    style={{ width: `${percentage}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Input Features Summary */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h4 className="text-sm font-semibold text-gray-700 mb-3">Input Features:</h4>
        <div className="grid grid-cols-2 gap-3 text-sm">
          {result.features && result.features.length === 4 && (
            <>
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-gray-600 text-xs mb-1">Sepal Length</p>
                <p className="font-bold text-gray-900">{result.features[0]} cm</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-gray-600 text-xs mb-1">Sepal Width</p>
                <p className="font-bold text-gray-900">{result.features[1]} cm</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-gray-600 text-xs mb-1">Petal Length</p>
                <p className="font-bold text-gray-900">{result.features[2]} cm</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-gray-600 text-xs mb-1">Petal Width</p>
                <p className="font-bold text-gray-900">{result.features[3]} cm</p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
