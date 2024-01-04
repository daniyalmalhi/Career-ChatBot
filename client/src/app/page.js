"use client";
import { useRouter } from "next/navigation";
import { BsArrowUpRight } from "react-icons/bs";
export default function Home() {
  const router = useRouter();
  return (
    <main className="w-full flex-center flex-col md:py-7 px-5 font-poppins md:flex md:flex-row justify-around items-center max-md:text-center max-md:w-full">
      <div>
        <h1 className="md:text-[6rem] text-[13vw] font-light mt-20">
          Introducing Chatbot
        </h1>
        <p className="md:text-[1.5rem] mb-5 md:max-w-[700px]">
        Success in creating AI would be the biggest event in human history.
        </p>
        <button
          className="flex items-center justify-center border-2 border-primaryBlue rounded-sm py-3 px-32 p-1 hover:bg-primaryBlue hover:text-white ease-in-out duration-100 max-md:mx-auto text-xl"
          onClick={() => router.push("/Chatroom")}
        >
          Try it out{" "}
          <span>
            <BsArrowUpRight />
          </span>
        </button>
      </div>
      <img
        src={"/assets/20943399-fotor-bg-remover-2023071422216.png"}
        alt="Landing Page Image"
        className="object-contain md:max-w-[500px] max-md:hidden"
      />
    </main>
  );
}
