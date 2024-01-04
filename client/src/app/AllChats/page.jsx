"use client";
import { useState, useEffect } from "react";
import Spinner from "@/components/Spinner";
import { format } from "date-fns";
import { BsTrash } from "react-icons/bs";
import Link from "next/link";

const AllChats = () => {
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleDelete = (index) => async () => {
    const res = await fetch(`/api/conversation/${index}`, {
      method: "DELETE",
    });
    if (res.ok) {
      window.location.reload();
    }
  };

  useEffect(() => {
    const id = localStorage.getItem("userID");
    const fetchConvo = async () => {
      setLoading(true);
      const res = await fetch(`/api/conversation/${id}`);
      const data = await res.json();
      setConversation(data);
      setLoading(false);
    };
    fetchConvo();
  }, []);

  return (
    <section className="flex flex-col items-center bg-primaryWhite shadow-md font-poppins py-5 mt-20 md:w-1/2 w-full">
      {conversation.length === 0 ? (
        <h1 className="text-3xl text-primaryBlue text-center font-bold">
          No Chats Created Yet :)
        </h1>
      ) : (
        <div>
          <h1 className="sm:text-5xl text-4xl mb-5 leading-none text-primaryBlue text-center">
            All Chats
          </h1>
          {loading ? (
            <div className="flex items-center justify-center">
              <Spinner />
            </div>
          ) : (
            conversation.map((convo, index) => (
              <div
                key={index}
                className="flex items-center py-3 text-primaryBlue w-full sm:px-20 px-3 mx-auto bg-blue-100 my-3 shadow-md"
              >
                <div className="sm:mr-10">
                  <h1 className="sm:text-xl text-lg">{convo.title}</h1>
                  <p className="text-sm text-gray-500">
                    Last Edited: {format(new Date(convo.date), "dd MMM yyyy")}
                  </p>
                </div>
                <div className="flex items-center gap-x-5 ml-auto">
                  <Link href={`/Chatroom/${convo._id}`}>
                    <button className="text-md bg-slate-700 hover:bg-slate-600 py-1 px-5 rounded-md text-white">
                      View
                    </button>
                  </Link>
                  <BsTrash
                    color="darkred"
                    className="cursor-pointer"
                    onClick={handleDelete(convo._id)}
                  />
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </section>
  );
};

export default AllChats;
