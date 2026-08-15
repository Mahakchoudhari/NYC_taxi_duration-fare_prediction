import { useState } from "react";
import "./App.css";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
  useMap,
} from "react-leaflet";

// Leaflet marker icon fix
import "leaflet/dist/leaflet.css";

import L from "leaflet";

delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",

  iconUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",

  shadowUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});
// Map Component

function TaxiMap({ formData }) {
  const pickup = [
    Number(formData.pickup_latitude),
    Number(formData.pickup_longitude),
  ];

  const dropoff = [
    Number(formData.dropoff_latitude),
    Number(formData.dropoff_longitude),
  ];

  const center = [
    (pickup[0] + dropoff[0]) / 2,
    (pickup[1] + dropoff[1]) / 2,
  ];

  return (
    <MapContainer
      center={center}
      zoom={13}
      scrollWheelZoom={true}
      className="taxi-map"
    >
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      <Marker position={pickup}>
        <Popup>
          🟢 <strong>Pickup</strong>
          <br />
          {pickup[0].toFixed(4)}, {pickup[1].toFixed(4)}
        </Popup>
      </Marker>

      <Marker position={dropoff}>
        <Popup>
          🔴 <strong>Dropoff</strong>
          <br />
          {dropoff[0].toFixed(4)}, {dropoff[1].toFixed(4)}
        </Popup>
      </Marker>

      <Polyline
        positions={[pickup, dropoff]}
      />
    </MapContainer>
  );
}
function App() {
  // =====================================================
  // DEFAULT FORM VALUES
  // =====================================================

  const initialFormData = {
    vendor_id: "1",

    pickup_date: "2016-06-12",
    pickup_time: "08:30",

    passenger_count: "1",

    pickup_latitude: "40.7489",
    pickup_longitude: "-73.9680",

    dropoff_latitude: "40.7614",
    dropoff_longitude: "-73.9776",

    store_and_fwd_flag: "N",
  };

  // =====================================================
  // STATES
  // =====================================================

  const [formData, setFormData] = useState(initialFormData);

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  // =====================================================
  // HANDLE INPUT CHANGE
  // =====================================================

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));

    // Remove previous error when user changes input
    if (error) {
      setError("");
    }
  };

  // =====================================================
  // RESET
  // =====================================================

  const handleReset = () => {
    setFormData(initialFormData);
    setResult(null);
    setError("");
  };

  // =====================================================
  // PREDICT
  // =====================================================

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      // -------------------------------------------------
      // CREATE DATETIME
      // -------------------------------------------------

      const pickupDatetime =
        `${formData.pickup_date}T${formData.pickup_time}`;

      // -------------------------------------------------
      // CONVERT NUMBERS
      // -------------------------------------------------

      const vendorId = Number(formData.vendor_id);

      const passengerCount =
        Number(formData.passenger_count);

      const pickupLatitude =
        Number(formData.pickup_latitude);

      const pickupLongitude =
        Number(formData.pickup_longitude);

      const dropoffLatitude =
        Number(formData.dropoff_latitude);

      const dropoffLongitude =
        Number(formData.dropoff_longitude);

      // -------------------------------------------------
      // VALIDATE NUMBERS
      // -------------------------------------------------

      if (!Number.isFinite(vendorId)) {
        throw new Error(
          "Vendor ID is invalid."
        );
      }

      if (
        !Number.isFinite(passengerCount) ||
        passengerCount < 1
      ) {
        throw new Error(
          "Passenger count must be at least 1."
        );
      }

      if (
        !Number.isFinite(pickupLatitude) ||
        pickupLatitude < -90 ||
        pickupLatitude > 90
      ) {
        throw new Error(
          "Pickup latitude must be between -90 and 90."
        );
      }

      if (
        !Number.isFinite(pickupLongitude) ||
        pickupLongitude < -180 ||
        pickupLongitude > 180
      ) {
        throw new Error(
          "Pickup longitude must be between -180 and 180."
        );
      }

      if (
        !Number.isFinite(dropoffLatitude) ||
        dropoffLatitude < -90 ||
        dropoffLatitude > 90
      ) {
        throw new Error(
          "Dropoff latitude must be between -90 and 90."
        );
      }

      if (
        !Number.isFinite(dropoffLongitude) ||
        dropoffLongitude < -180 ||
        dropoffLongitude > 180
      ) {
        throw new Error(
          "Dropoff longitude must be between -180 and 180."
        );
      }

      // -------------------------------------------------
      // CREATE PAYLOAD
      // -------------------------------------------------

      const payload = {
        vendor_id: vendorId,

        pickup_datetime: pickupDatetime,

        passenger_count: passengerCount,

        pickup_longitude: pickupLongitude,
        pickup_latitude: pickupLatitude,

        dropoff_longitude: dropoffLongitude,
        dropoff_latitude: dropoffLatitude,

        store_and_fwd_flag:
          formData.store_and_fwd_flag,
      };

      // -------------------------------------------------
      // DEBUG LOG
      // -------------------------------------------------

      console.log(
        "===================================="
      );

      console.log(
        "🚕 NYC TAXI PREDICTION REQUEST"
      );

      console.log(
        "API URL:",
        "http://127.0.0.1:8000/predict"
      );

      console.log(
        "Payload:",
        payload
      );

      console.log(
        "JSON Payload:",
        JSON.stringify(
          payload,
          null,
          2
        )
      );

      console.log(
        "===================================="
      );

      // -------------------------------------------------
      // SEND REQUEST
      // -------------------------------------------------

      let response;

      try {
        response = await fetch(
          "http://127.0.0.1:8000/predict",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",

              Accept:
                "application/json",
            },

            body: JSON.stringify(
              payload
            ),
          }
        );
      } catch (networkError) {
        // ---------------------------------------------
        // NETWORK / CORS ERROR
        // ---------------------------------------------

        console.error(
          "===================================="
        );

        console.error(
          "❌ NETWORK / CORS ERROR"
        );

        console.error(
          "Error:",
          networkError
        );

        console.error(
          "Message:",
          networkError?.message
        );

        console.error(
          "===================================="
        );

        throw new Error(
          `Unable to connect to FastAPI.

API:
http://127.0.0.1:8000/predict

Browser error:
${networkError?.message || "Failed to fetch"}

Possible causes:
• FastAPI is not running
• CORS problem
• Wrong port
• Backend is unreachable

Open browser Console for the complete error.`
        );
      }

      // -------------------------------------------------
      // RESPONSE STATUS
      // -------------------------------------------------

      console.log(
        "📡 HTTP STATUS:",
        response.status
      );

      console.log(
        "📡 STATUS TEXT:",
        response.statusText
      );

      console.log(
        "📡 RESPONSE URL:",
        response.url
      );

      console.log(
        "📡 RESPONSE HEADERS:",
        [...response.headers.entries()]
      );

      // -------------------------------------------------
      // READ RAW RESPONSE
      // -------------------------------------------------

      const responseText =
        await response.text();

      console.log(
        "📦 RAW BACKEND RESPONSE:"
      );

      console.log(
        responseText
      );

      // -------------------------------------------------
      // PARSE RESPONSE
      // -------------------------------------------------

      let data;

      try {
        data =
          JSON.parse(responseText);
      } catch (jsonError) {
        console.error(
          "❌ JSON PARSE ERROR:",
          jsonError
        );

        throw new Error(
          `Backend returned invalid JSON.

HTTP Status:
${response.status}

Raw Response:
${responseText}`
        );
      }

      console.log(
        "📦 PARSED BACKEND RESPONSE:",
        data
      );

      // -------------------------------------------------
      // HTTP ERROR
      // -------------------------------------------------

      if (!response.ok) {
        console.error(
          "===================================="
        );

        console.error(
          "❌ BACKEND HTTP ERROR"
        );

        console.error(
          "Status:",
          response.status
        );

        console.error(
          "Response:",
          data
        );

        console.error(
          "===================================="
        );

        let backendError =
          data?.detail ||
          data?.error ||
          data?.message ||
          "Unknown backend error.";

        // FastAPI validation error
        if (
          Array.isArray(
            data?.detail
          )
        ) {
          backendError =
            data.detail
              .map((item) => {
                return (
                  item?.msg ||
                  JSON.stringify(item)
                );
              })
              .join("\n");
        }

        throw new Error(
          `FastAPI returned HTTP ${response.status}.

${backendError}`
        );
      }

      // -------------------------------------------------
      // SUCCESS FALSE
      // -------------------------------------------------

      if (
        data.success === false
      ) {
        console.error(
          "❌ PREDICTION FAILED:"
        );

        console.error(
          data
        );

        throw new Error(
          data.error ||
          "Backend returned success=false."
        );
      }

      // -------------------------------------------------
      // CHECK RESPONSE
      // -------------------------------------------------

      if (
        data.duration_minutes ===
        undefined
      ) {
        console.error(
          "❌ UNEXPECTED BACKEND RESPONSE:"
        );

        console.error(
          data
        );

        throw new Error(
          `Backend response does not contain duration_minutes.

Received:
${JSON.stringify(
  data,
  null,
  2
)}`
        );
      }

      // -------------------------------------------------
      // SUCCESS
      // -------------------------------------------------

      console.log(
        "===================================="
      );

      console.log(
        "🎉 PREDICTION SUCCESS"
      );

      console.log(
        "Duration:",
        data.duration_minutes
      );

      console.log(
        "Fare:",
        data.estimated_fare
      );

      console.log(
        "Distance:",
        data.distance_km
      );

      console.log(
        "Speed:",
        data.estimated_speed
      );

      console.log(
        "Complete result:",
        data
      );

      console.log(
        "===================================="
      );

      setResult(data);

    } catch (err) {
      // -------------------------------------------------
      // FINAL ERROR
      // -------------------------------------------------

      console.error(
        "===================================="
      );

      console.error(
        "❌ FINAL PREDICTION ERROR"
      );

      console.error(
        "Error message:",
        err?.message
      );

      console.error(
        "Full error:",
        err
      );

      console.error(
        "===================================="
      );

      setError(
        err?.message ||
        "Something went wrong."
      );

    } finally {
      setLoading(false);
    }
  };

  // =====================================================
  // JSX
  // =====================================================

  return (
    <div className="app">

      {/* ==========================================
          NAVBAR
      ========================================== */}

      <nav className="navbar">

        <div className="logo">
          🚕 NYC<span>Ride</span>
        </div>

        <div className="nav-links">

          <a href="#predict">
            Predict
          </a>

          <a href="#results">
            Results
          </a>

          <a href="#model">
            Model
          </a>

        </div>

      </nav>


      {/* ==========================================
          HERO
      ========================================== */}

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
            Estimate trip duration, fare,
            distance and average speed using
            a machine learning model trained
            on NYC taxi trip data.
          </p>

        </div>

      </section>


      {/* ==========================================
          MAIN
      ========================================== */}

      <main className="container">


        {/* ==========================================
            FORM
        ========================================== */}

        <section
          className="card"
          id="predict"
        >

          <div className="card-header">

            <div>

              <h2>
                Trip Details
              </h2>

              <p>
                Enter your pickup and
                dropoff information
              </p>

            </div>

            <span className="icon">
              📍
            </span>

          </div>


          <form
            onSubmit={handleSubmit}
          >

            {/* ======================================
                PICKUP DATE + TIME
            ====================================== */}

            <div className="grid">

              <div className="form-group">

                <label>
                  Pickup Date
                </label>

                <input
                  type="date"
                  name="pickup_date"
                  value={
                    formData.pickup_date
                  }
                  onChange={
                    handleChange
                  }
                  required
                />

              </div>


              <div className="form-group">

                <label>
                  Pickup Time
                </label>

                <input
                  type="time"
                  name="pickup_time"
                  value={
                    formData.pickup_time
                  }
                  onChange={
                    handleChange
                  }
                  required
                />

              </div>

            </div>


            {/* ======================================
                PICKUP LOCATION
            ====================================== */}

            <div className="location-section">

              <h3>
                🟢 Pickup Location
              </h3>

              <div className="grid">

                <div className="form-group">

                  <label>
                    Latitude
                  </label>

                  <input
                    type="number"
                    step="0.0001"
                    min="-90"
                    max="90"
                    name="pickup_latitude"
                    value={
                      formData.pickup_latitude
                    }
                    onChange={
                      handleChange
                    }
                    required
                  />

                </div>


                <div className="form-group">

                  <label>
                    Longitude
                  </label>

                  <input
                    type="number"
                    step="0.0001"
                    min="-180"
                    max="180"
                    name="pickup_longitude"
                    value={
                      formData.pickup_longitude
                    }
                    onChange={
                      handleChange
                    }
                    required
                  />

                </div>

              </div>

            </div>


            {/* ======================================
                DROPOFF LOCATION
            ====================================== */}

            <div className="location-section">

              <h3>
                🔴 Dropoff Location
              </h3>

              <div className="grid">

                <div className="form-group">

                  <label>
                    Latitude
                  </label>

                  <input
                    type="number"
                    step="0.0001"
                    min="-90"
                    max="90"
                    name="dropoff_latitude"
                    value={
                      formData.dropoff_latitude
                    }
                    onChange={
                      handleChange
                    }
                    required
                  />

                </div>


                <div className="form-group">

                  <label>
                    Longitude
                  </label>

                  <input
                    type="number"
                    step="0.0001"
                    min="-180"
                    max="180"
                    name="dropoff_longitude"
                    value={
                      formData.dropoff_longitude
                    }
                    onChange={
                      handleChange
                    }
                    required
                  />

                </div>

              </div>

            </div>


            {/* ======================================
                OTHER DETAILS
            ====================================== */}

            <div className="grid">

              {/* VENDOR */}

              <div className="form-group">

                <label>
                  Vendor
                </label>

                <select
                  name="vendor_id"
                  value={
                    formData.vendor_id
                  }
                  onChange={
                    handleChange
                  }
                >

                  <option value="1">
                    Vendor 1
                  </option>

                  <option value="2">
                    Vendor 2
                  </option>

                </select>

              </div>


              {/* PASSENGERS */}

              <div className="form-group">

                <label>
                  Passengers
                </label>

                <select
                  name="passenger_count"
                  value={
                    formData.passenger_count
                  }
                  onChange={
                    handleChange
                  }
                >

                  {[1, 2, 3, 4, 5, 6].map(
                    (num) => (

                      <option
                        key={num}
                        value={num}
                      >
                        {num}
                      </option>

                    )
                  )}

                </select>

              </div>


              {/* STORE FORWARD */}

              <div className="form-group">

                <label>
                  Store & Forward
                </label>

                <select
                  name="store_and_fwd_flag"
                  value={
                    formData.store_and_fwd_flag
                  }
                  onChange={
                    handleChange
                  }
                >

                  <option value="N">
                    No
                  </option>

                  <option value="Y">
                    Yes
                  </option>

                </select>

              </div>

            </div>


            {/* ======================================
                BUTTONS
            ====================================== */}

            <div className="button-group">

              <button
                type="submit"
                className="predict-btn"
                disabled={loading}
              >

                {loading
                  ? "⏳ Predicting..."
                  : "🔮 Predict Trip"}

              </button>


              <button
                type="button"
                className="reset-btn"
                onClick={
                  handleReset
                }
                disabled={loading}
              >
                Reset
              </button>

            </div>

          </form>


          {/* ======================================
              ERROR
          ====================================== */}

          {error && (

            <div className="error-message">

              <strong>
                ❌ Prediction Error
              </strong>

              <pre>
                {error}
              </pre>

            </div>

          )}

        </section>
        <section className="map-section">

  <div className="section-title">

    <h2>
      Trip Route
    </h2>

    <p>
      Pickup and dropoff locations
    </p>

  </div>

  <div className="map-card">

    <TaxiMap formData={formData} />

  </div>

</section>

        {/* ==========================================
            RESULTS
        ========================================== */}

        <section
          className="result-section"
          id="results"
        >

          <div className="section-title">

            <h2>
              Prediction Results
            </h2>

            <p>
              AI-generated estimates
              for your trip
            </p>

          </div>


          <div className="result-grid">


            {/* DURATION */}

            <div className="result-card">

              <div className="result-icon">
                ⏱️
              </div>

              <p>
                Duration
              </p>

              <h3>
                {result
                  ? result.duration_minutes
                  : "--"}
              </h3>

              <span>
                minutes
              </span>

            </div>


            {/* FARE */}

            <div className="result-card">

              <div className="result-icon">
                💵
              </div>

              <p>
                Estimated Fare
              </p>

              <h3>
                {result
                  ? `$${result.estimated_fare}`
                  : "--"}
              </h3>

              <span>
                USD
              </span>

            </div>


            {/* DISTANCE */}

            <div className="result-card">

              <div className="result-icon">
                📍
              </div>

              <p>
                Distance
              </p>

              <h3>
                {result
                  ? result.distance_km
                  : "--"}
              </h3>

              <span>
                kilometers
              </span>

            </div>


            {/* SPEED */}

            <div className="result-card">

              <div className="result-icon">
                🚀
              </div>

              <p>
                Average Speed
              </p>

              <h3>
                {result
                  ? result.estimated_speed
                  : "--"}
              </h3>

              <span>
                km/h
              </span>

            </div>

          </div>

        </section>


        {/* ==========================================
            MODEL INFORMATION
        ========================================== */}

        <section
          className="model-section"
          id="model"
        >

          <div>

            <span className="small-label">
              MACHINE LEARNING MODEL
            </span>

            <h2>
              XGBoost Regression
            </h2>

            <p>
              The prediction system uses
              a tuned XGBoost model trained
              on more than 1.45 million
              NYC taxi trips with engineered
              temporal and geographical
              features.
            </p>

          </div>


          <div className="model-stats">

            <div>

              <strong>
                1.45M+
              </strong>

              <span>
                Trips
              </span>

            </div>


            <div>

              <strong>
                XGBoost
              </strong>

              <span>
                Algorithm
              </span>

            </div>


            <div>

              <strong>
                FastAPI
              </strong>

              <span>
                Backend
              </span>

            </div>

          </div>

        </section>

      </main>


      {/* ==========================================
          FOOTER
      ========================================== */}

      <footer>

        <p>
          NYC Ride • AI-Powered Taxi
          Prediction System
        </p>

      </footer>

    </div>
  );
}

export default App;