import React, { useState } from "react";
import { supabase } from "../api/supabase";

function AuthPage({ setCurrentUser }) {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    if (!email || !password) {
      alert("Please fill all required fields");
      setLoading(false);
      return;
    }

    try {
      let result;

      if (isSignUp) {
        // ✅ Sign Up
        result = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: {
              first_name: firstName,
              last_name: lastName,
            },
          },
        });
      } else {
        // ✅ Sign In
        result = await supabase.auth.signInWithPassword({
          email,
          password,
        });
      }

      if (result.error) {
        alert(result.error.message);
      } else if (result.data?.user) {
        const user = {
          email: result.data.user.email,
          firstName: result.data.user.user_metadata?.first_name || "",
          lastName: result.data.user.user_metadata?.last_name || "",
        };
        localStorage.setItem("currentUser", JSON.stringify(user));
        setCurrentUser(user);
      }
    } catch (err) {
      console.error("Auth error:", err);
      alert("Something went wrong");
    }

    setLoading(false);
  };

  return (
    <div id="auth-page" className="page active">
      <div className="auth-background">
        <div className="auth-decorations">
          <div className="decoration decoration-1"></div>
          <div className="decoration decoration-2"></div>
          <div className="decoration decoration-3"></div>
        </div>

        <div className="auth-card">
          <div className="auth-header">
            <h1>{isSignUp ? "Create Account" : "Welcome Back"}</h1>
            <p>
              {isSignUp
                ? "Sign up to start analyzing your data with AI"
                : "Sign in to your AI Data Analyst account"}
            </p>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            {isSignUp && (
              <div className="name-fields">
                <div className="form-group half">
                  <label>First Name</label>
                  <input
                    type="text"
                    value={firstName}
                    onChange={(e) => setFirstName(e.target.value)}
                  />
                </div>
                <div className="form-group half">
                  <label>Last Name</label>
                  <input
                    type="text"
                    value={lastName}
                    onChange={(e) => setLastName(e.target.value)}
                  />
                </div>
              </div>
            )}

            <div className="form-group">
              <label>Email Address</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label>Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              {isSignUp && <small>Minimum 6 characters required</small>}
            </div>

            <button type="submit" className="auth-button" disabled={loading}>
              {loading
                ? "Processing..."
                : isSignUp
                ? "Create Account"
                : "Sign In"}
            </button>

            <button
              type="button"
              className="toggle-auth"
              onClick={() => setIsSignUp(!isSignUp)}
            >
              {isSignUp
                ? "Already have an account? Sign in"
                : "Don't have an account? Sign up"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default AuthPage;
