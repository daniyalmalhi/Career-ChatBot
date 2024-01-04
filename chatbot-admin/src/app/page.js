"use client";
import { useState} from "react";
import { signIn } from "next-auth/react";
import Spinner from "@/components/Spinner";
import { useRouter } from "next/navigation";

const LogIn = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await signIn('credentials', {
        email,
        password,
        redirect: false
      });
      console.log(res)
      if (res.error) {
        console.log(res.error)
        setLoading(false)
        setError("Invalid Credentials");
        setResponse(null);
        return;
      }

      if(res.ok){
        setLoading(false)
        setResponse("Logged In Successfully");
        router.replace('/Dashboard')
      }

    } catch (error) {
      setLoading(false)
      console.log(error)
    }
  }

  return (
    <section className="flex flex-col items-center bg-primaryWhite shadow-md p-4 px-32 font-poppins mt-20">
      <h1 className="text-3xl font-bold my-5">Log In</h1>
      <form className="flex flex-col gap-2" onSubmit={handleSubmit}>
      <label htmlFor="email" className="mr-auto">
        Username
      </label>
      <input
        type="text"
        className="input_signIn"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <label htmlFor="password" className="mr-auto">
        Password
      </label>
      <input
        type="password"
        className="input_signIn"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      {loading ? <button className="btn mt-5"><div className="flex items-center justify-center"><Spinner/></div></button> : <button className="btn my-5">Log in</button>}
      {error && <p className="text-red-500 text-center mt-5 border-[1.5px] border-red-800 bg-rose-100 p-3">{error}</p>}
      {response && <p className="text-green-500 text-center mt-5 border-[1.5px] border-green-800 bg-green-50 p-3">{response}</p>}
      </form>
    </section>
  );
};

export default LogIn;
