import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./style.css";

// The only place React touches the real DOM: everything else renders into this root.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
