export default function Header() {
  return (
    <header className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-white/20 backdrop-blur-sm rounded-lg p-2">
              <span className="text-3xl">🌸</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold">Iris Species Classifier</h1>
              <p className="text-blue-100 text-sm">AI-Powered Flower Recognition</p>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-6">
            <div className="text-right">
              <p className="text-xs text-blue-100">Powered by</p>
              <p className="font-semibold">Machine Learning</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
