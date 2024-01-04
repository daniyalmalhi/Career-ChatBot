"use client";
import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { AiOutlineMenu } from "react-icons/ai";
import { AiOutlineClose } from "react-icons/ai";
import { BsArrowUpRight } from "react-icons/bs";
import { MdAccountCircle } from "react-icons/md";
import { useSession, signOut } from "next-auth/react";
import { usePathname, useRouter } from "next/navigation";
import { AiOutlinePlus } from "react-icons/ai";
import { PiChatsCircleFill } from "react-icons/pi";
import { CiSettings } from "react-icons/ci";
import { BsTrash, BsFillTelephoneFill } from "react-icons/bs";
import { BiLogOutCircle } from "react-icons/bi";
import Image from "next/image";
import { toast } from "react-toastify";
import Spinner from "./Spinner";

const Navbar = () => {
  const pathname = usePathname();
  const router = useRouter();

  let [open, setOpen] = useState(false);
  const { data: session, status } = useSession();
  const [toggledropdown, settoggledropdown] = useState(false);

  const handleClearConvo = async () => {
    localStorage.removeItem("conversation");
    localStorage.removeItem("context");
    const res = await fetch("http://localhost:8080/trash");
    const data = await res.json();
    //check the current url
    if (pathname !== "/Chatroom") {
      router.push("/Chatroom");
    } else {
      window.location.reload();
    }
    settoggledropdown((prev) => !prev);
  };

  const handleNewChat = () => {
    //check the current url
    if (pathname !== "/Chatroom") {
      router.push("/Chatroom");
    } else {
      window.location.reload();
    }
    settoggledropdown((prev) => !prev);
  };

  useEffect(() => {
    const handleOutsideClick = (e) => {
      // Check if the click is outside the dropdown and close it
      if (toggledropdown && !dropdownRef.current.contains(e.target)) {
        settoggledropdown(false);
      }
    };

    document.addEventListener("click", handleOutsideClick);

    return () => {
      document.removeEventListener("click", handleOutsideClick);
    };
  }, [toggledropdown]);
  const dropdownRef = useRef(null);
  if(status === "loading"){
    return (
      <div className="shadow-md fixed top-0 left-0 w-full z-10">
      <div className="bg-primaryBlue md:flex py-10 items-center justify-between text-white">
        </div>
      </div>
    )
  }
  return (
    <div className="shadow-md fixed top-0 left-0 w-full z-10">
      <div className="bg-primaryBlue md:flex py-4 items-center justify-between text-white">
        <Link href={"/"}>
          <div className="text-3xl font-normal font-poppins px-4 ">Chatbot</div>
        </Link>
        <div
          className="absolute top-5 right-0 text-xl items-center md:hidden px-2 cursor-pointer"
          onClick={() => setOpen((prev) => !prev)}
        >
          {open ? <AiOutlineClose size={30} /> : <AiOutlineMenu size={30} />}
        </div>

        {!session?.user ? (
          <div
            className={`md:flex md:flex-row-reverse items-center md-max:justify-center md:px-4 md:py-1 py-3 md:space-y-0 text-center absolute md:static bg-primaryBlue text-white md:z-10 w-full md:w-auto transition-all ease-in-out ${
              open ? "opacity-95" : "top-[-409px] md:opacity-100 opacity-0"
            }`}
          >
            <MdAccountCircle
              size={40}
              className="m-2 my-2 cursor-pointer mx-auto"
              onClick={() => settoggledropdown((prev) => !prev)}
            />
            {toggledropdown && (
              <div className="dropdown" ref={dropdownRef}>
                <Link
                  href="/Chatroom"
                  className="dropdown_link"
                  onClick={() => handleClearConvo()}
                >
                  <div className="flex items-center gap-2">
                    <span>
                      <BsTrash />
                    </span>
                    Clear Conversation
                  </div>
                </Link>
                <Link
                  className="dropdown_link"
                  href='/Formula'
                >
                  
                </Link>
              </div>
            )}
            <button
              onClick={() => router.push("/SignIn")}
              className="md:mr-5 py-2 px-5 md:bg-primaryWhite md:text-primaryBlue rounded-sm space-x-1 md:hover:bg-slate-100 md:flex md:items-center max-md:hover:text-slate-400"
            >
              Sign Up
              <span className="max-md:hidden ml-2">
                <BsArrowUpRight />
              </span>
            </button>
            <button
              onClick={() => router.push("/LogIn")}
              className="md:mr-5 py-2 px-5 md:bg-primaryWhite md:text-primaryBlue rounded-sm space-x-1 md:hover:bg-slate-100 md:flex md:items-center max-md:hover:text-slate-400"
            >
              Log In
              <span className="max-md:hidden ml-2">
                <BsArrowUpRight />
              </span>
            </button>
          </div>
        ) : (
          <div
            className={`md:flex md:flex-row-reverse items-center md-max:justify-center md:px-4 md:py-1 py-3 md:space-y-0 text-center absolute md:static bg-primaryBlue text-white md:z-10 w-full md:w-auto transition-all ease-in-out ${
              open ? "opacity-95" : "top-[-409px] md:opacity-100 opacity-0"
            }`}
          >
            {session?.user.provider === "credentials" ? (
              <MdAccountCircle
                size={37}
                onClick={() => settoggledropdown((prev) => !prev)}
                className="cursor-pointer max-md:mx-auto mt-5 md:mt-0"
              />
            ) : (
              <Image
                src={session?.user.image}
                width={37}
                height={37}
                className="rounded-full cursor-pointer max-md:mx-auto mt-5 md:mt-0"
                alt="profile"
                onClick={() => settoggledropdown((prev) => !prev)}
              />
            )}
            {toggledropdown && (
              <div className="dropdown" ref={dropdownRef}>
                <div
                  className="dropdown_link cursor-pointer"
                  onClick={()=>{
                    handleNewChat()
                    handleClearConvo()
                  }}
                >
                  <div className="flex items-center gap-2">
                    <span>
                      <AiOutlinePlus />
                    </span>
                    New Chat
                  </div>
                </div>
                <Link
                  href="/AllChats"
                  className="dropdown_link"
                  onClick={() => settoggledropdown(false)}
                >
                  <div className="flex items-center gap-2">
                    <span>
                      <PiChatsCircleFill />
                    </span>
                    All Chat
                  </div>
                </Link>
                <Link
                  className="dropdown_link"
                  href={`http://localhost:3001/?id=${session?.user.id}`}
                  target="blank"
                >
                  <div className="flex items-center gap-2">
                    <span>
                      <BsFillTelephoneFill />
                    </span>
                    Contact Professional User
                  </div>
                </Link>
               
                {session?.user.type === "Professional" && (
                  <Link
                    href="/Settings"
                    className="dropdown_link"
                    onClick={() => settoggledropdown(false)}
                  >
                    <div className="flex items-center gap-2">
                      <span>
                        <CiSettings />
                      </span>
                      Model Settings
                    </div>
                  </Link>
                )}

                <div
                  className="dropdown_link cursor-pointer"
                  onClick={() => handleClearConvo()}
                >
                  <div className="flex items-center gap-2">
                    <span>
                      <BsTrash />
                    </span>
                    Clear Conversation
                  </div>
                </div>
              </div>
            )}
            <button
              onClick={signOut}
              className="md:mr-5 py-2 px-5 md:bg-primaryWhite md:text-primaryBlue rounded-sm space-x-1 md:hover:bg-slate-100 md:flex md:items-center max-md:hover:text-slate-400"
            >
              Log Out
              <span className="ml-2 max-md:hidden">
                <BiLogOutCircle />
              </span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Navbar;
