import { useEffect, useRef, useState } from "react";
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

import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

/* =========================================================
   LEAFLET MARKER FIX
========================================================= */

delete L.Icon.Default.prototype._getIconUrl;

L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",

  iconUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",

  shadowUrl:
    "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});


/* =========================================================
   NYC SEARCH AREA
========================================================= */

const NYC_VIEWBOX = "-74.26,40.92,-73.68,40.49";


/* =========================================================
   ICON SYSTEM
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

    case "search":
      return (
        <svg {...common}>
          <circle cx="11" cy="11" r="7" />
          <path d="m20 20-3.2-3.2" />
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

    case "sliders":
      return (
        <svg {...common}>
          <path d="M4 6h10M18 6h2M4 12h2M8 12h12M4 18h14M20 18h0" />
          <circle cx="16" cy="6" r="2" />
          <circle cx="6" cy="12" r="2" />
          <circle cx="18" cy="18" r="2" />
        </svg>
      );

    case "chevron":
      return (
        <svg {...common}>
          <path d="m6 9 6 6 6-6" />
        </svg>
      );

    default:
      return null;
  }
}


/* =========================================================
   SPINNER
========================================================= */

function Spinner({ className = "" }) {
  return (
    <span
      className={`spinner ${className}`}
      aria-hidden="true"
    />
  );
}


/* =========================================================
   LOCATION SEARCH
========================================================= */

function LocationSearch({
  id,
  label,
  placeholder,
  tone,
  query,
  onQueryChange,
  onSelect,
}) {

  const [suggestions, setSuggestions] =
    useState([]);

  const [open, setOpen] =
    useState(false);

  const [searching, setSearching] =
    useState(false);

  const debounceRef =
    useRef(null);

  const wrapRef =
    useRef(null);


  useEffect(() => {

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    if (!query || query.trim().length < 3) {

      setSuggestions([]);
      setSearching(false);

      return;
    }

    debounceRef.current =
      setTimeout(async () => {

        setSearching(true);

        try {

          const url =
            `https://nominatim.openstreetmap.org/search?format=json&addressdetails=0` +
            `&limit=5&countrycodes=us&viewbox=${NYC_VIEWBOX}&bounded=1` +
            `&q=${encodeURIComponent(query)}`;

          const res =
            await fetch(url);

          const data =
            await res.json();

          setSuggestions(
            Array.isArray(data)
              ? data
              : []
          );

        } catch (err) {

          console.error(
            "Location search error:",
            err
          );

          setSuggestions([]);

        } finally {

          setSearching(false);

        }

      }, 450);


    return () =>
      clearTimeout(
        debounceRef.current
      );

  }, [query]);


  useEffect(() => {

    const handleClickOutside = (e) => {

      if (
        wrapRef.current &&
        !wrapRef.current.contains(e.target)
      ) {

        setOpen(false);

      }

    };

    document.addEventListener(
      "mousedown",
      handleClickOutside
    );

    return () =>
      document.removeEventListener(
        "mousedown",
        handleClickOutside
      );

  }, []);


  const shortLabel = (displayName) => {

    const parts =
      displayName.split(",");

    return parts
      .slice(0, 2)
      .join(",")
      .trim();

  };


  const handleSelect = (item) => {

    onQueryChange(
      shortLabel(
        item.display_name
      )
    );

    onSelect(
      Number(item.lat),
      Number(item.lon)
    );

    setOpen(false);
    setSuggestions([]);

  };


  return (

    <div
      className="location-search"
      ref={wrapRef}
    >

      <label htmlFor={id}>
        {label}
      </label>

      <div
        className={`location-search-box tone-${tone}`}
      >

        <Icon
          name="pin"
          className={`icon-inline icon-${tone}`}
        />

        <input
          id={id}
          type="text"
          autoComplete="off"
          placeholder={placeholder}
          value={query}
          onChange={(e) => {

            onQueryChange(
              e.target.value
            );

            setOpen(true);

          }}
          onFocus={() =>
            suggestions.length > 0 &&
            setOpen(true)
          }
        />

        {searching ? (

          <Spinner />

        ) : (

          <Icon
            name="search"
            className="icon-inline search-hint"
          />

        )}

      </div>


      {open &&
        suggestions.length > 0 && (

          <ul className="location-suggestions">

            {suggestions.map(
              (item) => (

                <li
                  key={item.place_id}
                  onMouseDown={() =>
                    handleSelect(item)
                  }
                >

                  <Icon
                    name="pin"
                    className="icon-inline"
                  />

                  {item.display_name}

                </li>

              )
            )}

          </ul>

        )}


      {open &&
        !searching &&
        query.trim().length >= 3 &&
        suggestions.length === 0 && (

          <ul className="location-suggestions">

            <li className="no-results">
              No matches in NYC — try a different search
            </li>

          </ul>

        )}

    </div>
  );
}


/* =========================================================
   MAP AUTO FIT
========================================================= */

function FitRoute({
  route,
  pickup,
  dropoff,
}) {

  const map = useMap();

  useEffect(() => {

    const bounds =
      route.length > 0
        ? L.latLngBounds(route)
        : L.latLngBounds([
            pickup,
            dropoff,
          ]);

    map.fitBounds(
      bounds,
      {
        padding: [40, 40],
      }
    );

  }, [
    map,
    route,
    pickup,
    dropoff,
  ]);

  return null;
}


/* =========================================================
   TAXI MAP
========================================================= */

function TaxiMap({
  formData,
}) {

  const [route, setRoute] =
    useState([]);

  const [routeDistance, setRouteDistance] =
    useState(null);

  const [routeDuration, setRouteDuration] =
    useState(null);

  const [routeLoading, setRouteLoading] =
    useState(false);

  const [routeError, setRouteError] =
    useState("");


  const pickup = [
    Number(
      formData.pickup_latitude
    ),
    Number(
      formData.pickup_longitude
    ),
  ];


  const dropoff = [
    Number(
      formData.dropoff_latitude
    ),
    Number(
      formData.dropoff_longitude
    ),
  ];


  const center = [
    (pickup[0] + dropoff[0]) / 2,
    (pickup[1] + dropoff[1]) / 2,
  ];


  const getRoute = async () => {

    setRouteLoading(true);
    setRouteError("");

    try {

      const url =
        `https://router.project-osrm.org/route/v1/driving/` +
        `${pickup[1]},${pickup[0]};${dropoff[1]},${dropoff[0]}` +
        `?overview=full&geometries=geojson`;

      const response =
        await fetch(url);

      if (!response.ok) {

        throw new Error(
          "Unable to fetch road route."
        );

      }


      const data =
        await response.json();


      if (
        !data.routes ||
        data.routes.length === 0
      ) {

        throw new Error(
          "No route found between these locations."
        );

      }


      const selectedRoute =
        data.routes[0];


      const coordinates =
        selectedRoute.geometry.coordinates.map(
          ([longitude, latitude]) =>
            [
              latitude,
              longitude,
            ]
        );


      setRoute(
        coordinates
      );


      setRouteDistance(
        (
          selectedRoute.distance / 1000
        ).toFixed(2)
      );


      setRouteDuration(
        (
          selectedRoute.duration / 60
        ).toFixed(1)
      );


    } catch (error) {

      console.error(
        "Route error:",
        error
      );

      setRouteError(
        error.message ||
        "Unable to find route."
      );

    } finally {

      setRouteLoading(false);

    }

  };


  return (

    <div>

      <div className="route-controls">

        <button
          type="button"
          className="route-btn"
          onClick={getRoute}
          disabled={routeLoading}
        >

          {routeLoading ? (

            <Spinner className="spinner-dark" />

          ) : (

            <Icon name="route" />

          )}

          {routeLoading
            ? "Finding route"
            : "Show road route"}

        </button>


        {routeDistance && (

          <div className="route-info">

            <div>

              <strong>
                {routeDistance} km
              </strong>

              <span>
                Road distance
              </span>

            </div>


            <div>

              <strong>
                {routeDuration} min
              </strong>

              <span>
                Route time
              </span>

            </div>

          </div>

        )}

      </div>


      {routeError && (

        <div className="route-error">

          <Icon
            name="alert"
            className="icon-inline"
          />

          {routeError}

        </div>

      )}


      <MapContainer
        center={center}
        zoom={13}
        scrollWheelZoom={true}
        className="taxi-map"
        style={{
          height: "450px",
          width: "100%",
          minHeight: "450px",
        }}
      >

        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />


        <FitRoute
          route={route}
          pickup={pickup}
          dropoff={dropoff}
        />


        <Marker position={pickup}>

          <Popup>

            <strong>
              Pickup location
            </strong>

            <br />

            Latitude:
            {" "}
            {pickup[0].toFixed(4)}

            <br />

            Longitude:
            {" "}
            {pickup[1].toFixed(4)}

          </Popup>

        </Marker>


        <Marker position={dropoff}>

          <Popup>

            <strong>
              Dropoff location
            </strong>

            <br />

            Latitude:
            {" "}
            {dropoff[0].toFixed(4)}

            <br />

            Longitude:
            {" "}
            {dropoff[1].toFixed(4)}

          </Popup>

        </Marker>


        {route.length > 0 && (

          <Polyline
            positions={route}
            pathOptions={{
              weight: 6,
            }}
          />

        )}

      </MapContainer>

    </div>

  );
}


/* =========================================================
   TRIP INSIGHTS
========================================================= */

function getTripInsights(
  pickupDatetime,
  result
) {

  const date =
    new Date(
      pickupDatetime
    );

  const hour =
    date.getHours();

  const day =
    date.getDay();


  const isWeekend =
    day === 0 ||
    day === 6;


  const isRushHour =
    [
      7,
      8,
      9,
      10,
      16,
      17,
      18,
      19,
      20,
    ].includes(hour);


  const isNight =
    hour >= 22 ||
    hour <= 4;


  let tripType =
    "Short trip";


  if (result?.distance_km) {

    const distance =
      Number(
        result.distance_km
      );

    if (distance >= 10) {

      tripType =
        "Long trip";

    } else if (distance >= 5) {

      tripType =
        "Medium trip";

    }

  }


  let trafficStatus =
    "Normal traffic";


  if (isRushHour) {

    trafficStatus =
      "Peak hours";

  } else if (isNight) {

    trafficStatus =
      "Night hours";

  }


  return {
    hour,
    isWeekend,
    isRushHour,
    isNight,
    tripType,
    trafficStatus,
  };
}


/* =========================================================
   MAIN APP
========================================================= */

function App() {

  const [formData, setFormData] =
    useState({

      vendor_id: 1,

      pickup_date:
        "2016-06-12",

      pickup_time:
        "08:30",

      passenger_count:
        1,

      pickup_latitude:
        40.7489,

      pickup_longitude:
        -73.968,

      dropoff_latitude:
        40.7614,

      dropoff_longitude:
        -73.9776,

      store_and_fwd_flag:
        "N",

    });


  const [pickupQuery, setPickupQuery] =
    useState("");

  const [dropoffQuery, setDropoffQuery] =
    useState("");

  const [showCoords, setShowCoords] =
    useState(false);


  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");


  /* =======================================================
     ANALYTICS STATE
  ======================================================= */

  const [analyticsData, setAnalyticsData] =
    useState(null);

  const [analyticsLoading, setAnalyticsLoading] =
    useState(false);

  const [analyticsError, setAnalyticsError] =
    useState("");


  const pickupDatetime =
    `${formData.pickup_date}T${formData.pickup_time}`;


  const insights =
    getTripInsights(
      pickupDatetime,
      result
    );


  /* =======================================================
     FORM CHANGE
  ======================================================= */

  const handleChange = (e) => {

    const {
      name,
      value,
    } = e.target;


    setFormData({
      ...formData,
      [name]: value,
    });

  };


  /* =======================================================
     PICKUP SELECT
  ======================================================= */

  const handlePickupSelect =
    (lat, lon) => {

      setFormData(
        (prev) => ({
          ...prev,

          pickup_latitude:
            lat,

          pickup_longitude:
            lon,

        })
      );

    };


  /* =======================================================
     DROPOFF SELECT
  ======================================================= */

  const handleDropoffSelect =
    (lat, lon) => {

      setFormData(
        (prev) => ({
          ...prev,

          dropoff_latitude:
            lat,

          dropoff_longitude:
            lon,

        })
      );

    };


  /* =======================================================
     ANALYTICS API
  ======================================================= */

  const fetchAnalytics =
    async () => {

      setAnalyticsLoading(
        true
      );

      setAnalyticsError("");


      try {

        const response =
          await fetch(
            "http://127.0.0.1:8000/analytics"
          );


        const data =
          await response.json();


        if (!response.ok) {

          throw new Error(
            data.detail ||
            data.error ||
            "Failed to load analytics."
          );

        }


        if (!data.success) {

          throw new Error(
            data.error ||
            "Analytics request failed."
          );

        }


        setAnalyticsData(
          data
        );


      } catch (err) {

        console.error(
          "Analytics error:",
          err
        );


        setAnalyticsError(
          err.message ||
          "Unable to load analytics."
        );


      } finally {

        setAnalyticsLoading(
          false
        );

      }

    };


  /* =======================================================
     DOWNLOAD REPORT
  ======================================================= */

  const downloadReport =
    () => {

      if (!result) return;


      const report = `
========================================
        NYC RIDE - TRIP REPORT
========================================

TRIP DETAILS
----------------------------------------
Pickup Date & Time : ${pickupDatetime}

Pickup Location
${pickupQuery
  ? `Place             : ${pickupQuery}`
  : ""}
Latitude          : ${formData.pickup_latitude}
Longitude         : ${formData.pickup_longitude}

Dropoff Location
${dropoffQuery
  ? `Place             : ${dropoffQuery}`
  : ""}
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
Day Type           : ${
  insights.isWeekend
    ? "Weekend"
    : "Weekday"
}
Trip Time          : ${
  insights.isNight
    ? "Night"
    : "Daytime"
}
Trip Category      : ${insights.tripType}

========================================
Generated by NYC Ride
Machine Learning Prediction System
========================================
`;


      const blob =
        new Blob(
          [report],
          {
            type: "text/plain",
          }
        );


      const url =
        URL.createObjectURL(
          blob
        );


      const link =
        document.createElement(
          "a"
        );


      link.href =
        url;


      link.download =
        "NYC_Ride_Trip_Report.txt";


      document.body.appendChild(
        link
      );


      link.click();


      document.body.removeChild(
        link
      );


      URL.revokeObjectURL(
        url
      );

    };


  /* =======================================================
     PREDICTION
  ======================================================= */

  const handleSubmit =
    async (e) => {

      e.preventDefault();


      setLoading(true);

      setError("");

      setResult(null);


      try {

        const pickupLat =
          Number(
            formData.pickup_latitude
          );

        const pickupLon =
          Number(
            formData.pickup_longitude
          );

        const dropoffLat =
          Number(
            formData.dropoff_latitude
          );

        const dropoffLon =
          Number(
            formData.dropoff_longitude
          );


        if (
          !pickupQuery &&
          !showCoords
        ) {

          throw new Error(
            "Please search and select a pickup location."
          );

        }


        if (
          !dropoffQuery &&
          !showCoords
        ) {

          throw new Error(
            "Please search and select a dropoff location."
          );

        }


        if (
          Number.isNaN(
            pickupLat
          ) ||
          Number.isNaN(
            pickupLon
          )
        ) {

          throw new Error(
            "Pickup location is missing coordinates."
          );

        }


        if (
          Number.isNaN(
            dropoffLat
          ) ||
          Number.isNaN(
            dropoffLon
          )
        ) {

          throw new Error(
            "Dropoff location is missing coordinates."
          );

        }


        if (
          pickupLat < -90 ||
          pickupLat > 90
        ) {

          throw new Error(
            "Pickup latitude must be between -90 and 90."
          );

        }


        if (
          dropoffLat < -90 ||
          dropoffLat > 90
        ) {

          throw new Error(
            "Dropoff latitude must be between -90 and 90."
          );

        }


        if (
          pickupLon < -180 ||
          pickupLon > 180
        ) {

          throw new Error(
            "Pickup longitude must be between -180 and 180."
          );

        }


        if (
          dropoffLon < -180 ||
          dropoffLon > 180
        ) {

          throw new Error(
            "Dropoff longitude must be between -180 and 180."
          );

        }


        const response =
          await fetch(
            "http://127.0.0.1:8000/predict",
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body:
                JSON.stringify({

                  vendor_id:
                    Number(
                      formData.vendor_id
                    ),

                  pickup_datetime:
                    pickupDatetime,

                  passenger_count:
                    Number(
                      formData.passenger_count
                    ),

                  pickup_longitude:
                    pickupLon,

                  pickup_latitude:
                    pickupLat,

                  dropoff_longitude:
                    dropoffLon,

                  dropoff_latitude:
                    dropoffLat,

                  store_and_fwd_flag:
                    formData.store_and_fwd_flag,

                }),

            }
          );


        const data =
          await response.json();


        if (!response.ok) {

          throw new Error(
            data.detail ||
            data.error ||
            "Server error occurred."
          );

        }


        if (!data.success) {

          throw new Error(
            data.error ||
            "Prediction failed."
          );

        }


        setResult(
          data
        );


        setTimeout(
          () => {

            document
              .getElementById(
                "results"
              )
              ?.scrollIntoView({
                behavior:
                  "smooth",

                block:
                  "start",
              });

          },
          100
        );


      } catch (err) {

        console.error(
          "Prediction error:",
          err
        );


        setError(
          err.message ||
          "Unable to connect to prediction server."
        );


      } finally {

        setLoading(false);

      }

    };


  /* =======================================================
     ANALYTICS DATA SAFETY
  ======================================================= */

  const tripsByHour =
    analyticsData?.trips_by_hour ||
    [];

  const durationByHour =
    analyticsData?.duration_by_hour ||
    [];

  const vendorDistribution =
    analyticsData?.vendor_distribution ||
    [];


  const rushHourChartData = [

    {
      name: "Rush Hours",

      trips:
        analyticsData?.rush_hour
          ?.rush_trips || 0,
    },

    {
      name: "Normal Hours",

      trips:
        analyticsData?.rush_hour
          ?.normal_trips || 0,
    },

  ];


  const dayTypeChartData = [

    {
      name: "Weekday",

      value:
        analyticsData?.day_type
          ?.weekday_trips || 0,
    },

    {
      name: "Weekend",

      value:
        analyticsData?.day_type
          ?.weekend_trips || 0,
    },

  ];


  return (

    <div className="app">


      {/* =================================================
          NAVBAR
      ================================================= */}

      <nav className="navbar">

        <div className="logo">

          <Icon
            name="taxi"
            className="logo-icon"
          />

          NYC<span>Ride</span>

        </div>


        <div className="nav-links">

          <a href="#predict">
            Predict
          </a>

          <a href="#analytics">
            Analytics
          </a>

          <a href="#model">
            Model
          </a>

        </div>

      </nav>


      {/* =================================================
          HERO
      ================================================= */}

      <section className="hero">

        <div className="hero-content">

          <div className="badge">

            <Icon
              name="pulse"
              className="icon-inline"
            />

            AI-powered taxi prediction

          </div>


          <h1>

            Predict your{" "}

            <span>
              NYC taxi trip
            </span>

          </h1>


          <p>

            Estimate trip duration, fare,
            distance and average speed
            using a machine learning model
            trained on NYC taxi data.

          </p>

        </div>

      </section>


      {/* =================================================
          MAIN
      ================================================= */}

      <main
        className="container"
        id="predict"
      >


        {/* =================================================
            FORM
        ================================================= */}

        <section className="card">

          <div className="card-header">

            <div>

              <h2>
                Trip details
              </h2>

              <p>
                Search a pickup and dropoff
                spot — coordinates fill in
                automatically
              </p>

            </div>


            <span className="icon">

              <Icon name="pin" />

            </span>

          </div>


          <form
            onSubmit={
              handleSubmit
            }
          >


            {/* DATE + TIME */}

            <div className="grid">

              <div className="form-group">

                <label htmlFor="pickup_date">

                  <Icon
                    name="calendar"
                    className="icon-inline"
                  />

                  Pickup date

                </label>


                <input
                  id="pickup_date"
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

                <label htmlFor="pickup_time">

                  <Icon
                    name="clock"
                    className="icon-inline"
                  />

                  Pickup time

                </label>


                <input
                  id="pickup_time"
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


            {/* PICKUP */}

            <LocationSearch
              id="pickup_search"
              label="Pickup location"
              placeholder="Search an address, landmark or neighborhood…"
              tone="pickup"
              query={pickupQuery}
              onQueryChange={
                setPickupQuery
              }
              onSelect={
                handlePickupSelect
              }
            />


            {/* DROPOFF */}

            <LocationSearch
              id="dropoff_search"
              label="Dropoff location"
              placeholder="Search an address, landmark or neighborhood…"
              tone="dropoff"
              query={dropoffQuery}
              onQueryChange={
                setDropoffQuery
              }
              onSelect={
                handleDropoffSelect
              }
            />


            {/* MANUAL COORDINATES */}

            <button
              type="button"
              className="coords-toggle"
              onClick={() =>
                setShowCoords(
                  (v) => !v
                )
              }
            >

              <Icon
                name="sliders"
                className="icon-inline"
              />

              Enter exact coordinates instead

              <Icon
                name="chevron"
                className={`icon-inline chevron ${
                  showCoords
                    ? "open"
                    : ""
                }`}
              />

            </button>


            {showCoords && (

              <div className="location-section">

                <h3>
                  Manual coordinates
                </h3>


                <div className="grid">

                  <div className="form-group">

                    <label>
                      Pickup latitude
                    </label>

                    <input
                      type="number"
                      step="0.0001"
                      name="pickup_latitude"
                      value={
                        formData.pickup_latitude
                      }
                      onChange={
                        handleChange
                      }
                    />

                  </div>


                  <div className="form-group">

                    <label>
                      Pickup longitude
                    </label>

                    <input
                      type="number"
                      step="0.0001"
                      name="pickup_longitude"
                      value={
                        formData.pickup_longitude
                      }
                      onChange={
                        handleChange
                      }
                    />

                  </div>


                  <div className="form-group">

                    <label>
                      Dropoff latitude
                    </label>

                    <input
                      type="number"
                      step="0.0001"
                      name="dropoff_latitude"
                      value={
                        formData.dropoff_latitude
                      }
                      onChange={
                        handleChange
                      }
                    />

                  </div>


                  <div className="form-group">

                    <label>
                      Dropoff longitude
                    </label>

                    <input
                      type="number"
                      step="0.0001"
                      name="dropoff_longitude"
                      value={
                        formData.dropoff_longitude
                      }
                      onChange={
                        handleChange
                      }
                    />

                  </div>

                </div>

              </div>

            )}


            {/* OTHER DETAILS */}

            <div className="grid">

              <div className="form-group">

                <label>
                  Vendor ID
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

                  <option value={1}>
                    Vendor 1
                  </option>

                  <option value={2}>
                    Vendor 2
                  </option>

                </select>

              </div>


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


              <div className="form-group">

                <label>
                  Store & forward
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


            {/* ERROR */}

            {error && (

              <div className="error-message">

                <Icon
                  name="alert"
                  className="icon-inline"
                />

                {error}

              </div>

            )}


            {/* PREDICT */}

            <button
              className="predict-btn"
              type="submit"
              disabled={loading}
            >

              {loading ? (

                <Spinner />

              ) : (

                <Icon name="compass" />

              )}


              {loading
                ? "Predicting"
                : "Predict trip"}

            </button>

          </form>

        </section>


        {/* =================================================
            RESULTS
        ================================================= */}

        <section
          className="result-section"
          id="results"
        >

          <div className="section-title">

            <h2>
              Prediction results
            </h2>

            <p>
              AI-generated estimates
              for your trip
            </p>

          </div>


          {loading && (

            <div className="prediction-loading">

              <Spinner
                className="spinner-lg"
              />

              <p>
                Analyzing your trip
              </p>

              <span>
                Calculating duration,
                distance and fare
              </span>

            </div>

          )}


          <div className="result-grid">

            <div className="result-card">

              <Icon
                name="clock"
                className="result-icon"
              />

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


            <div className="result-card">

              <Icon
                name="dollar"
                className="result-icon"
              />

              <p>
                Estimated fare
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


            <div className="result-card">

              <Icon
                name="ruler"
                className="result-icon"
              />

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


            <div className="result-card">

              <Icon
                name="gauge"
                className="result-icon"
              />

              <p>
                Average speed
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


          {result && (

            <div className="report-container">

              <button
                type="button"
                className="report-btn"
                onClick={
                  downloadReport
                }
              >

                <Icon name="download" />

                Download trip report

              </button>

            </div>

          )}

        </section>


        {/* =================================================
            TRIP INSIGHTS
        ================================================= */}

        <section
          className="analytics-section"
        >

          <div className="section-title">

            <h2>
              Trip insights
            </h2>

            <p>
              Intelligent analysis
              based on your trip details
            </p>

          </div>


          <div className="analytics-grid">

            <div className="analytics-card">

              <div className="analytics-icon">

                <Icon name="traffic" />

              </div>

              <div>

                <span>
                  Traffic period
                </span>

                <h3>
                  {result
                    ? insights.trafficStatus
                    : "--"}
                </h3>

              </div>

            </div>


            <div className="analytics-card">

              <div className="analytics-icon">

                <Icon name="calendar" />

              </div>

              <div>

                <span>
                  Day type
                </span>

                <h3>
                  {result
                    ? (
                        insights.isWeekend
                          ? "Weekend"
                          : "Weekday"
                      )
                    : "--"}
                </h3>

              </div>

            </div>


            <div className="analytics-card">

              <div className="analytics-icon">

                <Icon name="clock" />

              </div>

              <div>

                <span>
                  Trip time
                </span>

                <h3>
                  {result
                    ? (
                        insights.isNight
                          ? "Night"
                          : "Daytime"
                      )
                    : "--"}
                </h3>

              </div>

            </div>


            <div className="analytics-card">

              <div className="analytics-icon">

                <Icon name="route" />

              </div>

              <div>

                <span>
                  Trip category
                </span>

                <h3>
                  {result
                    ? insights.tripType
                    : "--"}
                </h3>

              </div>

            </div>

          </div>


          {result && (

            <div className="insight-summary">

              <div>

                <strong>

                  <Icon
                    name="spark"
                    className="icon-inline"
                  />

                  AI trip analysis

                </strong>


                <p>

                  Your trip is classified
                  as a{" "}

                  <b>
                    {insights.tripType.toLowerCase()}
                  </b>{" "}

                  during{" "}

                  <b>
                    {insights.trafficStatus.toLowerCase()}
                  </b>.

                  The predicted journey
                  duration is{" "}

                  <b>
                    {result.duration_minutes}
                    {" "}
                    minutes
                  </b>{" "}

                  over approximately{" "}

                  <b>
                    {result.distance_km}
                    {" "}
                    km
                  </b>.

                </p>

              </div>

            </div>

          )}

        </section>


        {/* =================================================
            DATASET ANALYTICS
        ================================================= */}

        <section
          className="analytics-section"
          id="analytics"
        >

          <div className="section-title">

            <h2>
              NYC Taxi Analytics
            </h2>

            <p>
              Real statistics from the
              NYC taxi dataset
            </p>

          </div>


          {/* LOAD BUTTON */}

          {!analyticsData &&
            !analyticsLoading && (

              <div className="analytics-load">

                <button
                  type="button"
                  className="predict-btn"
                  onClick={
                    fetchAnalytics
                  }
                >

                  <Icon name="pulse" />

                  Load NYC Analytics

                </button>

              </div>

            )}


          {/* LOADING */}

          {analyticsLoading && (

            <div className="prediction-loading">

              <Spinner
                className="spinner-lg"
              />

              <p>
                Loading NYC taxi analytics
              </p>

              <span>
                Analyzing the dataset...
              </span>

            </div>

          )}


          {/* ERROR */}

          {analyticsError && (

            <div className="error-message">

              <Icon
                name="alert"
                className="icon-inline"
              />

              {analyticsError}

            </div>

          )}


          {/* ANALYTICS DASHBOARD */}

          {analyticsData && (

            <>

              {/* KPI CARDS */}

              <div className="analytics-kpi-grid">

                <div className="analytics-kpi">

                  <span>
                    Total Trips
                  </span>

                  <strong>
                    {analyticsData.total_trips
                      ? analyticsData.total_trips.toLocaleString()
                      : "--"}
                  </strong>

                  <small>
                    NYC taxi records
                  </small>

                </div>


                <div className="analytics-kpi">

                  <span>
                    Avg Duration
                  </span>

                  <strong>
                    {analyticsData.avg_duration_minutes ??
                      "--"}
                  </strong>

                  <small>
                    minutes
                  </small>

                </div>


                <div className="analytics-kpi">

                  <span>
                    Avg Passengers
                  </span>

                  <strong>
                    {analyticsData.avg_passengers ??
                      "--"}
                  </strong>

                  <small>
                    passengers / trip
                  </small>

                </div>


                <div className="analytics-kpi">

                  <span>
                    Rush Hour Trips
                  </span>

                  <strong>
                    {analyticsData.rush_hour?.rush_trips
                      ? analyticsData.rush_hour.rush_trips.toLocaleString()
                      : "--"}
                  </strong>

                  <small>
                    peak-hour trips
                  </small>

                </div>

              </div>


              {/* CHART ROW 1 */}

              <div className="analytics-chart-grid">


                {/* TRIPS BY HOUR */}

                <div className="analytics-chart-card">

                  <div className="chart-header">

                    <div>

                      <h3>
                        Trips by Hour
                      </h3>

                      <p>
                        Number of taxi trips
                        for each hour
                      </p>

                    </div>

                  </div>


                  <ResponsiveContainer
                    width="100%"
                    height={320}
                  >

                    <BarChart
                      data={
                        tripsByHour
                      }
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="hour"
                        label={{
                          value:
                            "Hour",
                          position:
                            "insideBottom",
                          offset:
                            -5,
                        }}
                      />

                      <YAxis />

                      <Tooltip />

                      <Bar
                        dataKey="trips"
                        radius={[
                          5,
                          5,
                          0,
                          0,
                        ]}
                      />

                    </BarChart>

                  </ResponsiveContainer>

                </div>


                {/* DURATION BY HOUR */}

                <div className="analytics-chart-card">

                  <div className="chart-header">

                    <div>

                      <h3>
                        Average Duration by Hour
                      </h3>

                      <p>
                        Average trip duration
                        throughout the day
                      </p>

                    </div>

                  </div>


                  <ResponsiveContainer
                    width="100%"
                    height={320}
                  >

                    <LineChart
                      data={
                        durationByHour
                      }
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="hour"
                      />

                      <YAxis />

                      <Tooltip />

                      <Line
                        type="monotone"
                        dataKey="duration_minutes"
                        strokeWidth={3}
                        dot={false}
                      />

                    </LineChart>

                  </ResponsiveContainer>

                </div>

              </div>


              {/* CHART ROW 2 */}

              <div className="analytics-chart-grid">


                {/* RUSH HOURS */}

                <div className="analytics-chart-card">

                  <div className="chart-header">

                    <div>

                      <h3>
                        Rush Hour Analysis
                      </h3>

                      <p>
                        Peak hours vs normal
                        hours
                      </p>

                    </div>

                  </div>


                  <ResponsiveContainer
                    width="100%"
                    height={320}
                  >

                    <BarChart
                      data={
                        rushHourChartData
                      }
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="name"
                      />

                      <YAxis />

                      <Tooltip />

                      <Bar
                        dataKey="trips"
                        radius={[
                          5,
                          5,
                          0,
                          0,
                        ]}
                      />

                    </BarChart>

                  </ResponsiveContainer>

                </div>


                {/* WEEKDAY VS WEEKEND */}

                <div className="analytics-chart-card">

                  <div className="chart-header">

                    <div>

                      <h3>
                        Weekday vs Weekend
                      </h3>

                      <p>
                        Trip distribution by
                        day type
                      </p>

                    </div>

                  </div>


                  <ResponsiveContainer
                    width="100%"
                    height={320}
                  >

                    <PieChart>

                      <Pie
                        data={
                          dayTypeChartData
                        }
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        dataKey="value"
                        label
                      >

                        <Cell />

                        <Cell />

                      </Pie>


                      <Tooltip />

                      <Legend />

                    </PieChart>

                  </ResponsiveContainer>

                </div>

              </div>


              {/* VENDOR DISTRIBUTION */}

              <div className="analytics-chart-card vendor-chart">

                <div className="chart-header">

                  <div>

                    <h3>
                      Vendor Distribution
                    </h3>

                    <p>
                      Number of trips handled
                      by each taxi vendor
                    </p>

                  </div>

                </div>


                <ResponsiveContainer
                  width="100%"
                  height={300}
                >

                  <BarChart
                    data={
                      vendorDistribution
                    }
                  >

                    <CartesianGrid
                      strokeDasharray="3 3"
                    />

                    <XAxis
                      dataKey="vendor_id"
                      tickFormatter={
                        (value) =>
                          `Vendor ${value}`
                      }
                    />

                    <YAxis />

                    <Tooltip />

                    <Bar
                      dataKey="trips"
                      radius={[
                        5,
                        5,
                        0,
                        0,
                      ]}
                    />

                  </BarChart>

                </ResponsiveContainer>

              </div>


              {/* REFRESH */}

              <div className="analytics-refresh">

                <button
                  type="button"
                  className="route-btn"
                  onClick={
                    fetchAnalytics
                  }
                  disabled={
                    analyticsLoading
                  }
                >

                  {analyticsLoading
                    ? "Refreshing..."
                    : "↻ Refresh analytics"}

                </button>

              </div>

            </>

          )}

        </section>


        {/* =================================================
            MAP
        ================================================= */}

        <section className="map-section">

          <div className="section-title">

            <h2>
              Trip route
            </h2>

            <p>
              Pickup and dropoff
              locations
            </p>

          </div>


          <div className="map-card">

            <TaxiMap
              formData={formData}
            />

          </div>

        </section>


        {/* =================================================
            MODEL INFO
        ================================================= */}

        <section
          className="model-section"
          id="model"
        >

          <div>

            <span className="small-label">
              Machine learning model
            </span>

            <h2>
              XGBoost regression
            </h2>

            <p>

              The prediction system uses
              a tuned XGBoost model trained
              on more than 1.45 million NYC
              taxi trips.

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
                Real-time
              </strong>

              <span>
                Inference
              </span>

            </div>

          </div>

        </section>

      </main>


      {/* =================================================
          FOOTER
      ================================================= */}

      <footer className="footer">

        <p>
          NYC Ride &middot;
          Machine learning prediction system
        </p>

      </footer>

    </div>

  );
}


export default App;