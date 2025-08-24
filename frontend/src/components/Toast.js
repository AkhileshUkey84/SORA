import { useEffect } from "react";

export default function Toast({ message, type = "info", onClose }) {
  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(onClose, 3000);
    return () => clearTimeout(timer);
  }, [message, onClose]);

  if (!message) return null;

  return (
    <div className={`toast toast-${type}`}>
      {message}
    </div>
  );
}
