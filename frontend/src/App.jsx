import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const videoRef = useRef(null);

  const [prediction, setPrediction] = useState("Waiting...");
  const [confidence, setConfidence] = useState(0);
  const [cameraOn, setCameraOn] = useState(false);

  // Start camera
  const startCamera = async () => {
    try {
      // If camera is already running, don't start another stream
      if (videoRef.current?.srcObject) {
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraOn(true);
      }
    } catch (error) {
      console.error("Camera error:", error);
      setCameraOn(false);
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject
        .getTracks()
        .forEach((track) => track.stop());

      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
  };

  // Stop camera when page/component is closed
  useEffect(() => {
    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject
          .getTracks()
          .forEach((track) => track.stop());
      }
    };
  }, []);

  // Test backend prediction
  const testPrediction = async () => {
    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
        }
      );

      const data = await response.json();

      setPrediction(data.text);
      setConfidence(data.confidence);
    } catch (error) {
      console.error("Prediction error:", error);
      setPrediction("Backend not connected");
      setConfidence(0);
    }
  };

  // Text-to-speech
  const speakText = (text) => {
    if (!text || text === "Waiting...") return;

    window.speechSynthesis.cancel();

    const speech = new SpeechSynthesisUtterance(text);
    speech.lang = "en-IN";
    speech.rate = 0.9;

    window.speechSynthesis.speak(speech);
  };

  return (
    <div className="app">

      <header className="header">
        <h1>🤟 ISL Communication Tool</h1>
        <p>Real-Time Indian Sign Language Recognition</p>
      </header>

      <main className="main-container">

        {/* Camera */}
        <section className="card camera-card">

          <h2>📷 Live Camera</h2>

          <div className="video-container">

            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
            />

            {!cameraOn && (
              <div className="camera-message">
                Camera is off
              </div>
            )}

          </div>

          <div className="button-row">

            <button
              className="primary-button"
              onClick={startCamera}
            >
              📷 Start Camera
            </button>

            <button
              className="stop-button"
              onClick={stopCamera}
            >
              🛑 Stop Camera
            </button>

            <button
              className="primary-button"
              onClick={testPrediction}
            >
              🔍 Test Recognition
            </button>

          </div>

        </section>

        {/* Recognition Result */}
        <section className="card result-card">

          <h2>Recognized Sign</h2>

          <div className="prediction">
            {prediction}
          </div>

          <div className="confidence">
            Confidence:{" "}
            <strong>
              {(confidence * 100).toFixed(1)}%
            </strong>
          </div>

          <button
            className="speak-button"
            onClick={() => speakText(prediction)}
          >
            🔊 Speak
          </button>

        </section>

      </main>

    </div>
  );
}

export default App;