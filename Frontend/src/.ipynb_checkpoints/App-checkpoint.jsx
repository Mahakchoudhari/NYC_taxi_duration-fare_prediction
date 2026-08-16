import { useEffect, useState } from "react";
import "./App.css";

import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
  useMap,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import L from "leaflet";

/* =========================================================
   LEAFLET MARKER FIX
========================================================= */

delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

/* =========================================================
   MAP AUTO FIT COMPONENT
========================================================= */

function FitRoute({ route, pickup, dropoff }) {
  const map = useMap();

  useEffect(() => {
    const bounds =
      route.length > 0 ? L.latLngBounds(route) : L.latLngBounds([pickup, dropoff]);

    map.fitBounds(bounds, { padding: [40, 40] });
  }, [map, route, pickup, dropoff]);

  return null;
}

/* =========================================================
   TAXI MAP
========================================================= */

function TaxiMap({ formData }) {
  const [route, setRoute] = useState([]);
  const [routeDistance, setRouteDistance] = useState(null);
  const [routeDuration, setRouteDuration] = useState(null);
  const [routeLoading, setRouteLoading] = useState(false);
  const [routeError, setRouteError] = useState("");

  const pickup = [Number(formData.pickup_latitude), Number(formData.pickup_longitude)];
  const dropoff = [Number(formData.dropoff_latitude), Number(formData.dropoff_longitude)];
  const center = [(pickup[0] + dropoff[0]) / 2, (pickup[1] + dropoff[1]) / 2];

  const getRoute = async () => {
    setRouteLoading(true);
    setRouteError("");

    try {
      const url =
        `https://router.project-osrm.org/route/v1/driving/` +
        `${pickup[1]},${pickup[0]};${dropoff[1]},${dropoff[0]}` +
        `?overview=full&geometries=geojson`;

      const response = await fetch(url);
      if (!response.ok) throw new Error("Unable to fetch road route.");

      const data = await response.json();
      if (!data.routes || data.routes.length === 0) {
        throw new Error("No route found between these locations.");
      }

      const selectedRoute = data.routes[0];
      const coordinates = selectedRoute.geometry.coordinates.map(
        ([longitude, latitude]) => [latitude, longitude]
      );

      setRoute(coordinates);
      setRouteDistance((selectedRoute.distance / 1000).toFixed(2));
      setRouteDuration((selectedRoute.duration / 60).toFixed(1));
    } catch (error) {
      console.error("Route error:", error);
      setRouteError(error.message || "Unable to find route.");
    } finally {
      setRouteLoading(false);
    }
  };

  return (
    <div>
      {/* ROUTE CONTROLS */}
      <div className="route-controls">
        <button type="button" className="route-btn" onClick={getRoute} disabled={routeLoading}>
          {routeLoading ? "⏳ Finding Route..." : "🛣️ Show Road Route"}
        </button>

        {routeDistance && (
          <div className="route-info">
            <div>
              <strong>{routeDistance} km</strong>
              <span>Road Distance</span>
            </div>
            <div>
              <strong>{routeDuration} min</strong>
              <span>Route Time</span>
            </div>
          </div>
        )}
      </div>

      {routeError && <div className="route-error">❌ {routeError}</div>}

      {/* MAP */}
      <MapContainer
        center={center}
        zoom={13}
        scrollWheelZoom={true}
        className="taxi-map"
        style={{ height: "450px", width: "100%", minHeight: "450px" }}
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <FitRoute route={route} pickup={pickup} dropoff={dropoff} />

        <Marker position={pickup}>
          <Popup>
            <strong>🟢 Pickup Location</strong>
            <br />
            Latitude: {pickup[0].toFixed(4)}
            <br />
            Longitude: {pickup[1].toFixed(4)}
          </Popup>
        </Marker>

        <Marker position={dropoff}>
          <Popup>
            <strong>🔴 Dropoff Location</strong>
            <br />
            Latitude: {dropoff[0].toFixed(4)}
            <br />
            Longitude: {dropoff[1].toFixed(4)}
          </Popup>
        </Marker>

        {route.length > 0 && <Polyline positions={route} pathOptions={{ weight: 6 }} />}
      </MapContainer>
    </div>
  );
}

/* =========================================================
   TRIP INSIGHTS
========================================================= */

function getTripInsights(formData, result) {
  const date = new Date(formData.pickup_datetime);
  const hour = date.getHours();
  const day = date.getDay();

  const isWeekend = day === 0 || day === 6;
  const isRushHour = [7, 8, 9, 10, 16, 17, 18, 19, 20].includes(hour);
  const isNight = hour >= 22 || hour <= 4;

  let tripType = "Short Trip";
  if (result?.distance_km) {
    const distance = Number(result.distance_km);
    if (distance >= 10) tripType = "Long Trip";
    else if (distance >= 5) tripType = "Medium Trip";
  }

  let trafficStatus = "Normal Traffic";
  if (isRushHour) trafficStatus = "Peak Hours";
  else if (isNight) trafficStatus = "Night Hours";

  return { hour, isWeekend, isRushHour, isNight, tripType, trafficStatus };
}

/* =========================================================
   MAIN APP
========================================================= */

function App() {
  const [formData, setFormData] = useState({
    vendor_id: 1,
    pickup_datetime: "2016-06-12T08:30",
    passenger_count: 1,
    pickup_latitude: 40.7489,
    pickup_longitude: -73.968,
    dropoff_latitude: 40.7614,
    dropoff_longitude: -73.9776,
    store_and_fwd_flag: "N",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const insights = getTripInsights(formData, result);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  /* =====================================================
     DOWNLOAD REPORT (moved to component scope — this was
     the bug causing the white screen: it was previously
     declared inside handleSubmit, so the button's
     onClick={downloadReport} couldn't find it.)
  ===================================================== */

  const downloadReport = () => {
    if (!result) return;

    const report = `
========================================
        NYC RIDE - TRIP REPORT
========================================

TRIP DETAILS
----------------------------------------
Pickup Date & Time : ${formData.pickup_datetime}

Pickup Location
Latitude          : ${formData.pickup_latitude}
Longitude         : ${formData.pickup_longitude}

Dropoff Location
Latitude          : ${formData.dropoff_latitude}
Longitude         : ${formData.dropoff_longitude}

Vendor ID         : ${formData.vendor_id}
Passengers        : ${formData.passenger_count}
Store & Forward   : ${formData.store_and_fwd_flag}

PREDICTION RESULTS
----------------------------------------
Predicted Duration : ${result.duration_minutes} minutes
Estimated Fare     : $${result.estimated_fare}
Distance           : ${result.distance_km} km
Average Speed      : ${result.estimated_speed} km/h

TRIP INSIGHTS
----------------------------------------
Traffic Period     : ${insights.trafficStatus}
Day Type           : ${insights.isWeekend ? "Weekend" : "Weekday"}
Trip Time          : ${insights.isNight ? "Night" : "Daytime"}
Trip Category      : ${insights.tripType}

========================================
Generated by NYC Ride
Machine Learning Prediction System
========================================
`;

    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "NYC_Ride_Trip_Report.txt";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const pickupLat = Number(formData.pickup_latitude);
      const pickupLon = Number(formData.pickup_longitude);
      const dropoffLat = Number(formData.dropoff_latitude);
      const dropoffLon = Number(formData.dropoff_longitude);

      if (pickupLat < -90 || pickupLat > 90) {
        throw new Error("Pickup latitude must be between -90 and 90.");
      }
      if (dropoffLat < -90 || dropoffLat > 90) {
        throw new Error("Dropoff latitude must be between -90 and 90.");
      }
      if (pickupLon < -180 || pickupLon > 180) {
        throw new Error("Pickup longitude must be between -180 and 180.");
      }
      if (dropoffLon < -180 || dropoffLon > 180) {
        throw new Error("Dropoff longitude must be between -180 and 180.");
      }

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vendor_id: Number(formData.vendor_id),
          pickup_datetime: formData.pickup_datetime,
          passenger_count: Number(formData.passenger_count),
          pickup_longitude: pickupLon,
          pickup_latitude: pickupLat,
          dropoff_longitude: dropoffLon,
          dropoff_latitude: dropoffLat,
          store_and_fwd_flag: formData.store_and_fwd_flag,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.error || "Server error occurred.");
      }
      if (!data.success) {
        throw new Error(data.error || "Prediction failed.");
      }

      setResult(data);

      setTimeout(() => {
        document.getElementById("results")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 100);
    } catch (err) {
      console.error("Prediction error:", err);
      setError(err.message || "Unable to connect to prediction server.");
    } finally {
      setLoading(false);
    }
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
          <div className="badge">⚡ AI-Powered Taxi Prediction</div>
          <h1>
            Predict Your <span>NYC Taxi Trip</span>
          </h1>
          <p>
            Estimate trip duration, fare, distance and average speed using a
            machine learning model trained on NYC taxi data.
          </p>
        </div>
      </section>

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
            <div className="form-group full">
              <label>Pickup Date & Time</label>
              <input
                type="datetime-local"
                name="pickup_datetime"
                value={formData.pickup_datetime}
                onChange={handleChange}
                required
              />
            </div>

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
                    required
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
                    required
                  />
                </div>
              </div>
            </div>

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
                    required
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
                    required
                  />
                </div>
              </div>
            </div>

            <div className="grid">
              <div className="form-group">
                <label>Vendor ID</label>
                <select name="vendor_id" value={formData.vendor_id} onChange={handleChange}>
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

            {error && <div className="error-message">❌ {error}</div>}

            <button className="predict-btn" type="submit" disabled={loading}>
              {loading ? "⏳ Predicting..." : "🔮 Predict Trip"}
            </button>
          </form>
        </section>

        {/* RESULTS */}
        <section className="result-section" id="results">
          <div className="section-title">
            <h2>Prediction Results</h2>
            <p>AI-generated estimates for your trip</p>
          </div>

          {loading && (
            <div className="prediction-loading">
              <div className="loading-spinner"></div>
              <p>AI is analyzing your trip...</p>
              <span>Calculating duration, distance and fare</span>
            </div>
          )}

          <div className="result-grid">
            <div className="result-card">
              <div className="result-icon">⏱️</div>
              <p>Duration</p>
              <h3>{result ? result.duration_minutes : "--"}</h3>
              <span>minutes</span>
            </div>

            <div className="result-card">
              <div className="result-icon">💵</div>
              <p>Estimated Fare</p>
              <h3>{result ? `$${result.estimated_fare}` : "--"}</h3>
              <span>USD</span>
            </div>

            <div className="result-card">
              <div className="result-icon">📍</div>
              <p>Distance</p>
              <h3>{result ? result.distance_km : "--"}</h3>
              <span>kilometers</span>
            </div>

            <div className="result-card">
              <div className="result-icon">🚀</div>
              <p>Average Speed</p>
              <h3>{result ? result.estimated_speed : "--"}</h3>
              <span>km/h</span>
            </div>
          </div>

          {result && (
            <div className="report-container">
              <button type="button" className="report-btn" onClick={downloadReport}>
                📄 Download Trip Report
              </button>
            </div>
          )}
        </section>

        {/* TRIP INSIGHTS */}
        <section className="analytics-section" id="analytics">
          <div className="section-title">
            <h2>Trip Insights</h2>
            <p>Intelligent analysis based on your trip details</p>
          </div>

          <div className="analytics-grid">
            <div className="analytics-card">
              <div className="analytics-icon">🚦</div>
              <div>
                <span>Traffic Period</span>
                <h3>{result ? insights.trafficStatus : "--"}</h3>
              </div>
            </div>

            <div className="analytics-card">
              <div className="analytics-icon">📅</div>
              <div>
                <span>Day Type</span>
                <h3>{result ? (insights.isWeekend ? "Weekend" : "Weekday") : "--"}</h3>
              </div>
            </div>

            <div className="analytics-card">
              <div className="analytics-icon">🕐</div>
              <div>
                <span>Trip Time</span>
                <h3>{result ? (insights.isNight ? "Night" : "Daytime") : "--"}</h3>
              </div>
            </div>

            <div className="analytics-card">
              <div className="analytics-icon">🛣️</div>
              <div>
                <span>Trip Category</span>
                <h3>{result ? insights.tripType : "--"}</h3>
              </div>
            </div>
          </div>

          {result && (
            <div className="insight-summary">
              <div>
                <strong>🤖 AI Trip Analysis</strong>
                <p>
                  Your trip is classified as a{" "}
                  <b>{insights.tripType.toLowerCase()}</b> during{" "}
                  <b>{insights.trafficStatus.toLowerCase()}</b>. The predicted journey
                  duration is <b>{result.duration_minutes} minutes</b> over
                  approximately <b>{result.distance_km} km</b>.
                </p>
              </div>
            </div>
          )}
        </section>

        {/* MAP */}
        <section className="map-section">
          <div className="section-title">
            <h2>Trip Route</h2>
            <p>Pickup and dropoff locations</p>
          </div>
          <div className="map-card">
            <TaxiMap formData={formData} />
          </div>
        </section>

        {/* MODEL INFO */}
        <section className="model-section" id="model">
          <div>
            <span className="small-label">MACHINE LEARNING MODEL</span>
            <h2>XGBoost Regression</h2>
            <p>
              The prediction system uses a tuned XGBoost model trained on more
              than 1.45 million NYC taxi trips.
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
        <p>NYC Ride • Machine Learning Prediction System</p>
      </footer>
    </div>
  );
}

export default App;
