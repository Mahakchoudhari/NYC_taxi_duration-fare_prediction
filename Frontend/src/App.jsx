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
   ICON SET
   One small stroke-icon system instead of emoji, so every
   glyph shares weight, size and color with the rest of the UI.
========================================================= */

function Icon({ name, className = "" }) {
  const common = {
    className: `icon-svg ${className}`,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round",
    strokeLinejoin: "round",
  };

  switch (name) {
    case "taxi":
      return (
        <svg {...common}>
          <path d="M5 16h14l-1.4-5.6a2 2 0 0 0-1.94-1.4H8.34a2 2 0 0 0-1.94 1.4L5 16Z" />
          <path d="M7 16v2.2a.8.8 0 0 1-.8.8H5a1 1 0 0 1-1-1v-2" />
          <path d="M17 16v2.2a.8.8 0 0 0 .8.8H19a1 1 0 0 0 1-1v-2" />
          <circle cx="8" cy="16.5" r="1.3" />
          <circle cx="16" cy="16.5" r="1.3" />
          <path d="M9 9V6.5A1.5 1.5 0 0 1 10.5 5h3A1.5 1.5 0 0 1 15 6.5V9" />
        </svg>
      );
    case "pulse":
      return (
        <svg {...common}>
          <path d="M3 12h4l2 7 4-14 2 7h6" />
        </svg>
      );
    case "pin":
      return (
        <svg {...common}>
          <path d="M12 21s7-6.1 7-11.5A7 7 0 0 0 5 9.5C5 14.9 12 21 12 21Z" />
          <circle cx="12" cy="9.5" r="2.4" />
        </svg>
      );
    case "route":
      return (
        <svg {...common}>
          <circle cx="6" cy="19" r="2" />
          <circle cx="18" cy="5" r="2" />
          <path d="M8 19h7a3 3 0 0 0 3-3v-1a3 3 0 0 0-3-3H9a3 3 0 0 1-3-3V8a3 3 0 0 1 3-3h7" />
        </svg>
      );
    case "compass":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="M14.8 9.2 13 13l-3.8 1.8L11 11l3.8-1.8Z" />
        </svg>
      );
    case "alert":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 8v5" />
          <path d="M12 16h.01" />
        </svg>
      );
    case "clock":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 7v5l3.5 2" />
        </svg>
      );
    case "dollar":
      return (
        <svg {...common}>
          <path d="M12 3v18" />
          <path d="M16.5 7.5a3.5 3.5 0 0 0-3.5-2H11a3 3 0 0 0 0 6h2a3 3 0 0 1 0 6h-2a3.5 3.5 0 0 1-3.5-2" />
        </svg>
      );
    case "ruler":
      return (
        <svg {...common}>
          <rect x="3" y="8" width="18" height="8" rx="1.5" />
          <path d="M7 8v3M11 8v3M15 8v3" />
        </svg>
      );
    case "gauge":
      return (
        <svg {...common}>
          <path d="M4 15a8 8 0 1 1 16 0" />
          <path d="M12 15l3.5-4.5" />
          <path d="M12 15h.01" />
        </svg>
      );
    case "download":
      return (
        <svg {...common}>
          <path d="M12 4v11" />
          <path d="M7.5 11.5 12 16l4.5-4.5" />
          <path d="M5 19h14" />
        </svg>
      );
    case "traffic":
      return (
        <svg {...common}>
          <rect x="8" y="3" width="8" height="14" rx="3" />
          <path d="M9 6h.01M12 9.5h.01M15 13h.01" />
          <path d="M11 20h2" />
        </svg>
      );
    case "calendar":
      return (
        <svg {...common}>
          <rect x="3.5" y="5" width="17" height="15" rx="2" />
          <path d="M3.5 10h17" />
          <path d="M8 3v4M16 3v4" />
        </svg>
      );
    case "spark":
      return (
        <svg {...common}>
          <path d="M12 3v4M12 17v4M3 12h4M17 12h4" />
          <path d="M6 6l2.5 2.5M17.5 15.5 20 18M18 6l-2.5 2.5M8.5 15.5 6 18" />
        </svg>
      );
    default:
      return null;
  }
}

function Spinner({ className = "" }) {
  return <span className={`spinner ${className}`} aria-hidden="true" />;
}

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
          {routeLoading ? <Spinner className="spinner-dark" /> : <Icon name="route" />}
          {routeLoading ? "Finding route" : "Show road route"}
        </button>

        {routeDistance && (
          <div className="route-info">
            <div>
              <strong>{routeDistance} km</strong>
              <span>Road distance</span>
            </div>
            <div>
              <strong>{routeDuration} min</strong>
              <span>Route time</span>
            </div>
          </div>
        )}
      </div>

      {routeError && (
        <div className="route-error">
          <Icon name="alert" className="icon-inline" />
          {routeError}
        </div>
      )}

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
            <strong>Pickup location</strong>
            <br />
            Latitude: {pickup[0].toFixed(4)}
            <br />
            Longitude: {pickup[1].toFixed(4)}
          </Popup>
        </Marker>

        <Marker position={dropoff}>
          <Popup>
            <strong>Dropoff location</strong>
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

  let tripType = "Short trip";
  if (result?.distance_km) {
    const distance = Number(result.distance_km);
    if (distance >= 10) tripType = "Long trip";
    else if (distance >= 5) tripType = "Medium trip";
  }

  let trafficStatus = "Normal traffic";
  if (isRushHour) trafficStatus = "Peak hours";
  else if (isNight) trafficStatus = "Night hours";

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
          <Icon name="taxi" className="logo-icon" />
          NYC<span>Ride</span>
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
            <Icon name="pulse" className="icon-inline" />
            AI-powered taxi prediction
          </div>
          <h1>
            Predict your <span>NYC taxi trip</span>
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
              <h2>Trip details</h2>
              <p>Enter your pickup and dropoff information</p>
            </div>
            <span className="icon">
              <Icon name="pin" />
            </span>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group full">
              <label>Pickup date & time</label>
              <input
                type="datetime-local"
                name="pickup_datetime"
                value={formData.pickup_datetime}
                onChange={handleChange}
                required
              />
            </div>

            <div className="location-section">
              <h3>
                <Icon name="pin" className="icon-inline icon-pickup" />
                Pickup location
              </h3>
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
              <h3>
                <Icon name="pin" className="icon-inline icon-dropoff" />
                Dropoff location
              </h3>
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
                <label>Store & forward</label>
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

            {error && (
              <div className="error-message">
                <Icon name="alert" className="icon-inline" />
                {error}
              </div>
            )}

            <button className="predict-btn" type="submit" disabled={loading}>
              {loading ? <Spinner /> : <Icon name="compass" />}
              {loading ? "Predicting" : "Predict trip"}
            </button>
          </form>
        </section>

        {/* RESULTS */}
        <section className="result-section" id="results">
          <div className="section-title">
            <h2>Prediction results</h2>
            <p>AI-generated estimates for your trip</p>
          </div>

          {loading && (
            <div className="prediction-loading">
              <Spinner className="spinner-lg" />
              <p>Analyzing your trip</p>
              <span>Calculating duration, distance and fare</span>
            </div>
          )}

          <div className="result-grid">
            <div className="result-card">
              <Icon name="clock" className="result-icon" />
              <p>Duration</p>
              <h3>{result ? result.duration_minutes : "--"}</h3>
              <span>minutes</span>
            </div>

            <div className="result-card">
              <Icon name="dollar" className="result-icon" />
              <p>Estimated fare</p>
              <h3>{result ? `$${result.estimated_fare}` : "--"}</h3>
              <span>USD</span>
            </div>

            <div className="result-card">
              <Icon name="ruler" className="result-icon" />
              <p>Distance</p>
              <h3>{result ? result.distance_km : "--"}</h3>
              <span>kilometers</span>
            </div>

            <div className="result-card">
              <Icon name="gauge" className="result-icon" />
              <p>Average speed</p>
              <h3>{result ? result.estimated_speed : "--"}</h3>
              <span>km/h</span>
            </div>
          </div>

          {result && (
            <div className="report-container">
              <button type="button" className="report-btn" onClick={downloadReport}>
                <Icon name="download" />
                Download trip report
              </button>
            </div>
          )}
        </section>

        {/* TRIP INSIGHTS */}
        <section className="analytics-section" id="analytics">
          <div className="section-title">
            <h2>Trip insights</h2>
            <p>Intelligent analysis based on your trip details</p>
          </div>

          <div className="analytics-grid">
            <div className="analytics-card">
              <div className="analytics-icon">
                <Icon name="traffic" />
              </div>
              <div>
                <span>Traffic period</span>
                <h3>{result ? insights.trafficStatus : "--"}</h3>
              </div>
            </div>

            <div className="analytics-card">
              <div className="analytics-icon">
                <Icon name="calendar" />
              </div>
              <div>
                <span>Day type</span>
                <h3>{result ? (insights.isWeekend ? "Weekend" : "Weekday") : "--"}</h3>
              </div>
            </div>

            <div className="analytics-card">
              <div className="analytics-icon">
                <Icon name="clock" />
              </div>
              <div>
                <span>Trip time</span>
                <h3>{result ? (insights.isNight ? "Night" : "Daytime") : "--"}</h3>
              </div>
            </div>

            <div className="analytics-card">
              <div className="analytics-icon">
                <Icon name="route" />
              </div>
              <div>
                <span>Trip category</span>
                <h3>{result ? insights.tripType : "--"}</h3>
              </div>
            </div>
          </div>

          {result && (
            <div className="insight-summary">
              <div>
                <strong>
                  <Icon name="spark" className="icon-inline" />
                  AI trip analysis
                </strong>
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
            <h2>Trip route</h2>
            <p>Pickup and dropoff locations</p>
          </div>
          <div className="map-card">
            <TaxiMap formData={formData} />
          </div>
        </section>

        {/* MODEL INFO */}
        <section className="model-section" id="model">
          <div>
            <span className="small-label">Machine learning model</span>
            <h2>XGBoost regression</h2>
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

      <footer className="footer">
        <p>NYC Ride &middot; Machine learning prediction system</p>
      </footer>
    </div>
  );
}

export default App;
