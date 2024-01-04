"use client";
import { useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import Spinner from "@/components/Spinner";

const Prompt = () => {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [url, setUrl] = useState("");
  const [success, setSuccess] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8080/prompt", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({question, answer, url}),
      });
      if (response.ok) {
        setLoading(false);
        setSuccess("Model Trained Successfully");
        setError(null);
        router.push("/Chatroom");
      } else {
        setLoading(false);
        setError("Something went wrong");
        setSuccess(null);
      }
    } catch (error) {
      console.log(error);
      setLoading(false);
      setError("Something went wrong");
      setSuccess(null);
    }
  };
  return (
    <section className="flex flex-col items-center bg-primaryWhite shadow-md font-poppins py-5 md:w-[35rem] mt-20">
      <div className="flex items-center  w-full justify-center">
        <Image
          src={"/assets/back-button.png"}
          width={30}
          height={30}
          className="object-contain mr-5 hover:cursor-pointer"
          alt="gobackbtn"
          onClick={() => router.push("/Settings")}
        />
        <h1 className="text-3xl font-normal">Insert by Prompt</h1>
      </div>
      <div className="border-[1px] border-black w-full m-4 mb-20" />
      <form className="w-full space-y-3 px-5" onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Question..."
          className="settings_input"
        />
        <textarea
          name="answer"
          id="answer"
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          className="settings_input"
          rows={3}
          placeholder="Answer..."
        ></textarea>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="URL References..."
          className="settings_input"
        />
        {loading ? <button className="btn"><div className="flex items-center justify-center"><Spinner/></div></button> : <button className="btn">Train Model</button>}
      </form>
      {error && (
        <p className="text-red-500 text-center mt-5 border-[1.5px] border-red-800 bg-rose-100 p-3">
          {error}
        </p>
      )}
      {success && (
        <p className="text-green-500 text-center mt-5 border-[1.5px] border-green-800 bg-green-50 p-3">
          {success}
        </p>
      )}
    </section>
  );
};

export default Prompt;
