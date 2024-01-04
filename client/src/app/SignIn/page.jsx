"use client";
import { useState, useEffect, Suspense } from "react";
import { FcGoogle } from "react-icons/fc";
import { FaFacebook } from "react-icons/fa";
import { FaGithub } from "react-icons/fa";
import { getProviders, signIn } from "next-auth/react";
import Spinner from "@/components/Spinner";
import { useRouter } from "next/navigation";

const SignIn = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState(null);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [providers, setProviders] = useState(null);
  const router = useRouter();

  useEffect(() => {
    const getProvidersList = async () => {
      const providers = await getProviders();
      setProviders(providers);
    };
    getProvidersList();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    if (password !== confirmPassword) {
      setLoading(false);
      setResponse(null);
      setError("Passwords do not match");
      return;
    }
    //other validations checks
    if (password.length < 8) {
      setLoading(false);
      setResponse(null);
      setError("Password should be atleast 8 characters long");
      return;
    }
    try {
      const res = await fetch("api/userExists", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email }),
      });

      const { user } = await res.json();

      if (user) {
        setLoading(false);
        setResponse(null);
        setError("User already exists");
        return;
      }
    } catch (error) {
      setLoading(false);
      console.log(error);
    }
    try {
      const res = await fetch("api/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });
      if (res.ok) {
        setLoading(false);
        setEmail("");
        setPassword("");
        setConfirmPassword("");
        setError(null);
        setResponse("User registered successfully");
        router.push("/LogIn");
      } else {
        setLoading(false);
        setError("User registration Failed");
        setResponse(null);
      }
    } catch (error) {
      setLoading(false);
      console.log(error);
    }
  };

  return (
    <section className="flex flex-col items-center bg-primaryWhite shadow-md p-4 px-12 font-poppins mt-20">
      <h1 className="text-3xl font-bold my-5">Sign Up</h1>
      <form className="flex flex-col gap-2" onSubmit={handleSubmit}>
        <label htmlFor="email" className="mr-auto">
          Email
        </label>
        <input
          type="email"
          className="input_signIn"
          placeholder="someone@email.com"
          value={email}
          required
          onChange={(e) => setEmail(e.target.value)}
        />
        <label htmlFor="password" className="mr-auto">
          Password
        </label>
        <input
          type="password"
          className="input_signIn"
          placeholder="password..."
          value={password}
          required
          onChange={(e) => setPassword(e.target.value)}
        />
        <label htmlFor="confirmpassword" className="mr-auto">
          Confirm Password
        </label>
        <input
          type="password"
          className="input_signIn"
          placeholder="confirm password..."
          value={confirmPassword}
          required
          onChange={(e) => setConfirmPassword(e.target.value)}
        />
        {error && (
          <p className="text-red-500 text-center mt-5 border-[1.5px] border-red-800 bg-rose-100 p-3">
            {error}
          </p>
        )}
        {response && (
          <p className="text-green-500 text-center mt-5 border-[1.5px] border-green-800 bg-green-50 p-3">
            {response}
          </p>
        )}
        {loading ? (
          <button className="btn mt-5">
            <div className="flex items-center justify-center">
              <Spinner />
            </div>
          </button>
        ) : (
          <button className="btn mt-5">Sign Up</button>
        )}
      </form>
      <h1 className="py-5">OR</h1>
      <button
        type="button"
        onClick={() => signIn("google", { callbackUrl: "/Chatroom" })}
        className="providers_btn mb-3"
      >
        <FcGoogle className="mr-3" />
        Sign in with Google
      </button>
      <button
        type="button"
        onClick={() => signIn("github", { callbackUrl: "/Chatroom" })}
        className="providers_btn mb-3"
      >
        <FaGithub className="mr-3" />
        Sign in with Github
      </button>
      <button
        type="button"
        onClick={() => signIn("facebook", { callbackUrl: "/Chatroom" })}
        className="providers_btn mb-3"
      >
        <FaFacebook className="mr-3" />
        Sign in with Facebook
      </button>
    </section>
  );
};

export default SignIn;
