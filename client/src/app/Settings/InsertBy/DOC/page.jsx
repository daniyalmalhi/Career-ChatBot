"use client";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState, useRef } from "react"; // import useRef for creating a reference to the file input
import Spinner from "@/components/Spinner";
import {toast} from "react-toastify";

const DOC = () => {
  const router = useRouter();
  const fileInputRef = useRef(null); // create a ref for the file input
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  const [loader, setLoader] = useState(false);

  // This would be a function triggered by the form submission.
  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoader(true);

    const file = fileInputRef.current.files[0]; // Get the file from the ref to the file input
    if (!file) {
      console.log("No file selected");
      return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("source", source);
    formData.append("file", file);

    console.log("formData", formData);

    try {
      // Post the form data to your Flask server
      const response = await fetch("http://localhost:8080/uploadpdf", {
        method: "POST",
        body: formData,
        // Headers are not included since FormData sets its own headers
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      toast.success("Document Uploaded Successfully");
      setLoader(false);
      console.log(result);
    } catch (error) {
      console.error("There was an error uploading the file", error);
      toast.error("There was an error uploading the file");
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
          alt="go back button"
          onClick={() => router.push("/Settings")}
        />
        <h1 className="text-3xl font-normal">Insert by Document</h1>
      </div>
      <div className="border-[1px] border-black w-full m-4 mb-20" />
      <form className="w-full space-y-3 px-5" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Document Name..."
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="settings_input"
        />
        <input
          ref={fileInputRef} // attach the ref here
          type="file"
          className="settings_input_file"
          accept=".pdf"
        />
        <input
          type="text"
          placeholder="Source"
          className="settings_input"
          value={source}
          onChange={(e) => setSource(e.target.value)}
        />
        {loader ? (
          <button className="btn flex items-center justify-center">
            <Spinner />
          </button>
        ) : (
          <button type="submit" className="btn">
            {" "}
            {/* Change this to type="submit" */}
            Upload Document
          </button>
        )}
      </form>
    </section>
  );
};

export default DOC;
