import { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    vendor_id: 1,
    pickup_datetime: "2016-06-12T08:30",
    passenger_count: 1,
    pickup_latitude: 40.7489,
    pickup_longitude: -73.9680,
    dropoff_latitude: 40.7614,
    dropoff_longitude: -73.9776,
    store_and_fwd_flag: "N",
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // API connection next step
    console.log(formData);

    setResult({
      duration_minutes: "--",
      estimated_fare: "--",
      distance_km: "--",
      estimated_speed: "--",
    });
  };

  return (
    <div className="app">

      {/* NAVBAR */}
      <nav className="navbar">
        <div className="logo">
          🚕 NYC<span>Ride</span>
        </div>

        <div className="nav-links">
          <a href="#predict">Predict</a>
          <a href="#analytics">Analytics</a>
          <a href="#model">Model</a>
        </div>
      </nav>


      {/* HERO */}
      <section className="hero">

        <div className="hero-content">

          <div className="badge">
            ⚡ AI-Powered Taxi Prediction
          </div>

          <h1>
            Predict Your
            <span> NYC Taxi Trip</span>
          </h1>

          <p>
            Estimate trip duration, fare, distance and average speed
            using a machine learning model trained on NYC taxi data.
          </p>

        </div>

      </section>


      {/* MAIN */}
      <main className="container" id="predict">

        {/* FORM CARD */}
        <section className="card">

          <div className="card-header">
            <div>
              <h2>Trip Details</h2>
              <p>Enter your pickup and dropoff information</p>
            </div>

            <span className="icon">📍</span>
          </div>


          <form onSubmit={handleSubmit}>

            {/* DATETIME */}
            <div className="form-group full">

              <label>Pickup Date & Time</label>

              <input
                type="datetime-local"
                name="pickup_datetime"
                value={formData.pickup_datetime}
                onChange={handleChange}
              />

            </div>


            {/* PICKUP */}
            <div className="location-section">

              <h3>🟢 Pickup Location</h3>

              <div className="grid">

                <div className="form-group">

                  <label>Latitude</label>

                  <input
                    type="number"
                    step="0.0001"
                    name="pickup_latitude"
                    value={formData.pickup_latitude}
                    onChange={handleChange}
                  />

                </div>

                <div className="form-group">

                  <label>Longitude</label>

                  <input
                    type="number"
                    step="0.0001"
                    name="pickup_longitude"
                    value={formData.pickup_longitude}
                    onChange={handleChange}
                  />

                </div>

              </div>

            </div>


            {/* DROPOFF */}
            <div className="location-section">

              <h3>🔴 Dropoff Location</h3>

              <div className="grid">

                <div className="form-group">

                  <label>Latitude</label>

                  <input
                    type="number"
                    step="0.0001"
                    name="dropoff_latitude"
                    value={formData.dropoff_latitude}
                    onChange={handleChange}
                  />

                </div>

                <div className="form-group">

                  <label>Longitude</label>

                  <input
                    type="number"
                    step="0.0001"
                    name="dropoff_longitude"
                    value={formData.dropoff_longitude}
                    onChange={handleChange}
                  />

                </div>

              </div>

            </div>


            {/* TRIP DETAILS */}
            <div className="grid">

              <div className="form-group">

                <label>Vendor ID</label>

                <select
                  name="vendor_id"
                  value={formData.vendor_id}
                  onChange={handleChange}
                >
                  <option value={1}>Vendor 1</option>
                  <option value={2}>Vendor 2</option>
                </select>

              </div>


              <div className="form-group">

                <label>Passengers</label>

                <select
                  name="passenger_count"
                  value={formData.passenger_count}
                  onChange={handleChange}
                >
                  {[1, 2, 3, 4, 5, 6].map((num) => (
                    <option key={num} value={num}>
                      {num}
                    </option>
                  ))}
                </select>

              </div>


              <div className="form-group">

                <label>Store & Forward</label>

                <select
                  name="store_and_fwd_flag"
                  value={formData.store_and_fwd_flag}
                  onChange={handleChange}
                >
                  <option value="N">No</option>
                  <option value="Y">Yes</option>
                </select>

              </div>

            </div>


            <button className="predict-btn" type="submit">
              🔮 Predict Trip
            </button>

          </form>

        </section>


        {/* RESULT */}
        <section className="result-section">

          <div className="section-title">

            <h2>Prediction Results</h2>

            <p>
              AI-generated estimates for your trip
            </p>

          </div>


          <div className="result-grid">

            <div className="result-card">

              <div className="result-icon">⏱️</div>

              <p>Duration</p>

              <h3>
                {result ? result.duration_minutes : "--"}
              </h3>

              <span>minutes</span>

            </div>


            <div className="result-card">

              <div className="result-icon">💵</div>

              <p>Estimated Fare</p>

              <h3>
                {result ? `$${result.estimated_fare}` : "--"}
              </h3>

              <span>USD</span>

            </div>


            <div className="result-card">

              <div className="result-icon">📍</div>

              <p>Distance</p>

              <h3>
                {result ? result.distance_km : "--"}
              </h3>

              <span>kilometers</span>

            </div>


            <div className="result-card">

              <div className="result-icon">🚀</div>

              <p>Average Speed</p>

              <h3>
                {result ? result.estimated_speed : "--"}
              </h3>

              <span>km/h</span>

            </div>

          </div>

        </section>


        {/* MODEL INFO */}
        <section className="model-section" id="model">

          <div>
            <span className="small-label">
              MACHINE LEARNING MODEL
            </span>

            <h2>XGBoost Regression</h2>

            <p>
              The prediction system uses a tuned XGBoost model
              trained on more than 1.45 million NYC taxi trips.
            </p>
          </div>

          <div className="model-stats">

            <div>
              <strong>1.45M+</strong>
              <span>Trips</span>
            </div>

            <div>
              <strong>XGBoost</strong>
              <span>Algorithm</span>
            </div>

            <div>
              <strong>Real-time</strong>
              <span>Inference</span>
            </div>

          </div>

        </section>

      </main>


      <footer>
        <p>
          NYC Ride • Machine Learning Prediction System
        </p>
      </footer>

    </div>
  );
}

export default App;