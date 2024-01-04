"use client";
import { useState, useEffect } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { GoVideo } from "react-icons/go";
import { BsFiletypeJson } from "react-icons/bs";
import { GoTerminal } from "react-icons/go";
import { IoDocumentsOutline } from "react-icons/io5";
import { GiPunch } from "react-icons/gi";

const Settings = () => {
  const router = useRouter();
  const [count, setCount] = useState(0);
  useEffect(() => {
    const fetchFeedbackCount = async () => {
      const res = await fetch("/api/feedback/get");
      const data = await res.json();
      setCount(data.length);
    };
    fetchFeedbackCount();
  }, []);
  return (
    <section className="flex flex-col items-center bg-primaryWhite shadow-md font-poppins py-5 md:w-[35rem] w-full mt-20">
      <div className="flex items-center  w-full justify-center">
        <Image
          src={"/assets/back-button.png"}
          width={30}
          height={30}
          className="object-contain mr-5 hover:cursor-pointer"
          alt="gobackbtn"
          onClick={() => router.push("/Chatroom")}
        />
        <h1 className="sm:text-3xl text-xl font-normal">Model Settings</h1>
      </div>
      <div className="border-[1px] border-black w-full m-4 mb-14" />
      <div className="w-full space-y-3 sm:px-[100px] px-5 mb-5">
        <div className="relative">
          <div className="px-3 py-1 bg-primaryBlue text-white rounded-full w-fit h-fit absolute inset-y-0 my-auto right-3 flex items-center justify-center">
            <p>{count}</p>
          </div>
          <button
            type="button"
            className="settings_btn"
            onClick={() => router.push("/Settings/Conflicts")}
          >
            <span>
              <GiPunch size={30} color="#164D5F" />
            </span>
            Review Feedback
          </button>
        </div>
        <button
          type="button"
          className="settings_btn"
          onClick={() => router.push("/Settings/InsertBy/Prompt")}
        >
          <span>
            <GoTerminal size={30} color="#164D5F" />
          </span>
          Insert by Prompt
        </button>
        <button
          type="button"
          className="settings_btn"
          onClick={() => router.push("/Settings/InsertBy/DOC")}
        >
          <span>
            <IoDocumentsOutline size={30} color="#164D5F" />
          </span>
          Insert by Document
        </button>
        <button
          type="button"
          className="settings_btn"
          onClick={() => router.push("/Settings/InsertBy/JSON")}
        >
          <span>
            <BsFiletypeJson size={30} color="#164D5F" />
          </span>
          Insert by JSON File
        </button>
       
      </div>
    </section>
  );
};

export default Settings;
