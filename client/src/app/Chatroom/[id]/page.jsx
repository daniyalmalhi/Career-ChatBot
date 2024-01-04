"use client";
import { useState, useEffect, useRef } from "react";
import { IoIosSend } from "react-icons/io";
import { format } from "date-fns";
import { MdAccountCircle } from "react-icons/md";
import { BiSolidBot } from "react-icons/bi";
import { useSession } from "next-auth/react";
import { AiOutlineLike } from "react-icons/ai";
import { AiOutlineDislike } from "react-icons/ai";
import { BsInfoCircle } from "react-icons/bs";
import { HiOutlineLightBulb } from "react-icons/hi";
import { BiCopy } from "react-icons/bi";
import { GoVerified } from "react-icons/go";
import { toast } from "react-toastify";
import { AiOutlineClose } from "react-icons/ai";
import Link from "next/link";
import Spinner from "@/components/Spinner";


import Image from "next/image";

const Chatroom = ({ params }) => {
  const [input, setInput] = useState("");
  const [context, setContext] = useState([]);
  const conversationContainerRef = useRef(null);
  const [isWaitingResponse, setIsWaitingResponse] = useState(false);
  const [conversation, setConversation] = useState([]);
  const { data: session, status } = useSession();
  const [copied, setCopied] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [source, setSource] = useState("No Links Selected.");
  const [sourceOpen, setSourceOpen] = useState(false);
  const [count, setCount] = useState(0);
  const [feedback, setFeedback] = useState({
    user: "",
    prompt: "",
    response: "",
    source: "",
    comments: "",
    thumbsUp: undefined,
  });
  const [feedbackLoader, setFeedbackLoader] = useState(false);

  const handleToggleSourceOpen = (source) => {
    if (source === undefined || source === null) {
      source = "No Links Selected.";
    }
    setSource(source);
    setSourceOpen(true);
  };

  const handleToggleSourceClose = () => {
    setSourceOpen(false);
  };

  const handleOpenModal = (prompt, response, thumbsUp) => {
    setFeedback({
      ...feedback,
      user: session?.user.id,
      prompt: prompt.text,
      response: response.text,
      source: response.source,
      thumbsUp: thumbsUp,
    });
    setModalOpen(true);
  };

  const handleFeedbackSubmit = async () => {
    setFeedbackLoader(true);
    if(feedback.comments === ''){
      toast.error("Please provide feedback");
      setFeedbackLoader(false);
      return;
    }
    await fetch("/api/feedback/post", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(feedback),
    }).then((response) => {
      if (response.ok) {
        setFeedbackLoader(false);
        toast.success("Feedback Submitted Successfully");
        setModalOpen(false);
        setFeedback({
          user: "",
          prompt: "",
          response: "",
          source: "",
          comments: "",
          thumbsUp: undefined,
        });
      } else {
        setFeedbackLoader(false);
        toast.error("Error submitting feedback");
      }
    });
  };

  const handleCloseModal = (event) => {
    event.stopPropagation();
    setFeedback({
      user: "",
      prompt: "",
      response: "",
      source: "",
      comments: "",
      thumbsUp: undefined,
    })
    setModalOpen(false);
  };

  useEffect(() => {
    // Prevent scrolling of body content when the modal is open
    if (modalOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "auto";
    }
  }, [modalOpen]);

  const handleCopy = (messageText) => {
    // Copy the message text to the clipboard
    navigator.clipboard
      .writeText(messageText)
      .then(() => {
        // Set the copied state to the message text
        setCopied(messageText);
        setTimeout(() => setCopied(false), 1000);
      })
      .catch((error) => {
        console.log("Error copying text to clipboard:", error);
      });
  };

  useEffect(() => {
    const fetchConvo = async () => {
      const res = await fetch(`/api/conversation/convo/${params.id}`);
      if (res.ok) {
        const data = await res.json();
        console.log(data[0]);
        setConversation(data[0].conversation);
        setCount(data[0].count);
        setContext(data[0].context);
      }
    };
    fetchConvo();
  }, []);

  useEffect(() => {
      if (context && context?.length > 0) {
        const updateContext = async () => {
          const res = await fetch('http://localhost:8080/updateContext', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({context: JSON.stringify(context)}
          )})
          if(res.ok){
            console.log('Context Updated')
          }
          else{
            console.log('Error Updating Context')
          }
        }
        updateContext()
      }
  }, [context]);

  useEffect(() => {
    localStorage.setItem("conversation", JSON.stringify(conversation));
    localStorage.setItem("context", JSON.stringify(context));
    localStorage.setItem("userID", session && session.user.id);
    // Scroll to the bottom of the conversation container when a new message is added
    if (conversationContainerRef.current) {
      conversationContainerRef.current.scrollTop =
        conversationContainerRef.current.scrollHeight;
    }
  }, [conversation, session]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsWaitingResponse(true);

    if (count === 20 && !session) {
      toast.info("Query Limit Reached, Please create an account to continue.");
      setIsWaitingResponse(false);
      return;
    }

    // Add the current input to the conversation
    setConversation([
      ...conversation,
      { text: input, isUser: true, source: "N/A", time: format(new Date(), "hh:mm a") },
    ]);
    setContext([...context, { role: "user", content: input }]);
    setInput("");

    try {
      // Send the user's input to the backend
      const response = await fetch("http://localhost:8080/chatbot", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify([context, { latestText: input }]), // Send the user's input as an object
      });
      if (response.ok) {
        const jsonData = await response.json();
        console.log(jsonData);
        let backendResponse = "";
        let source = "";
        if (jsonData[0].includes("Source of information:")) {
          backendResponse = jsonData[0].split("Source of information:")[0];
          source = jsonData[1].split("Source of information:")[1];
        } else if (
          jsonData[1].includes("https://") ||
          jsonData[1].includes("http://")
        ) {
          backendResponse = jsonData[0]; // Use the actual response from the backend
          source = jsonData[1].split("Source of information:")[1];
          if(!source){
            source = jsonData[1].split('https://')[1]
            if(source){
              source = 'https://' + source
            }
            else{
              source = jsonData[1].split('http://')[1]
              if(source){
                source = 'http://' + source
              }
              else{
                source = "No Source Provided.";
              }
            }
          }
        } else {
          backendResponse = jsonData[0]; // Use the actual response from the backend
          source = "No Source Provided.";
        }

        // Add the backend response to the conversation
        setConversation((prevConversation) => [
          ...prevConversation,
          {
            text: backendResponse,
            isUser: false,
            source: source,
            time: format(new Date(), "hh:mm a"),
          },
        ]);
        setContext((prevConversation) => [
          ...prevConversation,
          {
            role: "assistant",
            content: backendResponse,
          },
        ]);
        setCount((prev) => prev + 1);
        setIsWaitingResponse(false);
      } else {
        console.log("Error:", response.statusText);
      }
    } catch (error) {
      console.log("Error:", error);
    }
  };

  // This useEffect will trigger only when 'conversation' updates
  useEffect(() => {
    const user = session?.user.email;
    const storeconversation = async () => {
      if (conversation?.length > 0) {
        // Make a POST request to store 'conversation' in the database
        const res = await fetch("/api/conversation", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ conversation, context, count, user }),
        })
          .then((response) => {
            if (response.ok) {
              console.log("Conversation stored successfully");
            } else {
              console.log("Error storing conversation:", response.statusText);
            }
          })
          .catch((error) => {
            console.error("Error storing conversation:", error);
          });
      }
    };
    storeconversation();
  }, [count, session]);

  if(status === 'loading'){
    return(
      <div className="h-screen flex items-center justify-center">
        <Spinner/>
      </div>
    )
  }

  return (
    <div className="h-screen bg-white w-full font-poppins">
      <div
        className={`fixed inset-0 z-50 ${
          modalOpen ? "flex items-center justify-center" : "hidden"
        }`}
        style={{
          backgroundColor: "rgba(0, 0, 0, 0.2)", // Fallback background color for older browsers
          backdropFilter: modalOpen ? "blur(10px)" : "none", // Apply blur when modal is open
        }}
      >
        <div className="modal-container rounded-xl flex flex-col">
          <AiOutlineClose
            className="text-3xl cursor-pointer ml-auto relative left-8"
            color="black"
            onClick={handleCloseModal}
          />
          <h1
            className={`text-3xl text-center mb-10 ${
              feedback.thumbsUp === "like" ? "text-green-700" : "text-red-700"
            }`}
          >
            {feedback.thumbsUp === "like" ? "Thumbs Up" : "Thumbs Down"}
          </h1>
          <div>
            <textarea
              name="feedback"
              value={feedback.comments}
              required
              onChange={(e) =>
                setFeedback({ ...feedback, comments: e.target.value })
              }
              cols="100"
              rows="10"
              placeholder={`Provide Feedback for: '${feedback.response}'`}
              className={`border-2 p-2 w-full rounded-lg ${
                feedback.thumbsUp === "like"
                  ? "border-green-600 text-green-800"
                  : "border-red-600 text-red-800"
              }`}
            />

            {feedbackLoader ? (
              <button className="btn">
                <div className="flex items-center justify-center">
                  <Spinner />
                </div>
              </button>
            ) : (
              <button
                className={`w-full m-5 mx-auto border-2 p-2 rounded-lg ${
                  feedback.thumbsUp === "like"
                    ? "border-green-700 text-green-800 hover:bg-green-50"
                    : "border-red-700 text-red-800 hover:bg-red-50"
                }`}
                onClick={handleFeedbackSubmit}
              >
                Submit
              </button>
            )}
          </div>
        </div>
      </div>
      <div className="h-full flex flex-col justify-center items-center">
        {conversation?.length === 0 ? (
          // Default screen with example prompts
          <>
            <div className="md:text-[75px] text-[60px] mb-5 leading-none font-light text-primaryBlue text-center">
              Welcome to chatbot
            </div>
            <div className="text-3xl text-primaryBlue mb-10 text-center">
              Your AI-powered chatbot for something
            </div>
            <div className="flex items-center justify-center gap-2 text-xl mb-5">
              <span>
                <HiOutlineLightBulb />
              </span>
              Examples
            </div>
            <div className="grid md:grid-cols-2 grid-cols-1 sm:w-[700px] max-sm:w-full max-sm:px-10 gap-3">
              <div
                className="flex items-center justify-between text-xl bg-primaryBlue text-white px-6 py-3 rounded-md hover:bg-blue-50 hover:text-primaryBlue hover:border-[1.5px] hover:border-primaryBlue first-letter:duration-150 ease-in cursor-pointer"
                onClick={() => setInput("Example 1")}
              >
                Example 1
                {copied === "Example 1" ? (
                  <GoVerified
                    size={20}
                    className="object-contain transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                  />
                ) : (
                  <BiCopy
                    size={20}
                    className="object-contain transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                    onClick={() => handleCopy("Example 1")}
                  />
                )}
              </div>
              <div
                className="flex items-center justify-between text-xl bg-primaryBlue text-white px-6 py-3 rounded-md  hover:bg-blue-50 hover:text-primaryBlue hover:border-[1.5px] hover:border-primaryBlue first-letter:duration-150 ease-in cursor-pointer"
                onClick={() => setInput("Example 2")}
              >
                Example 2
                {copied === "Example 2" ? (
                  <GoVerified
                    size={20}
                    className="object-contain transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                  />
                ) : (
                  <BiCopy
                    size={20}
                    className="object-contain transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                    onClick={() => handleCopy("Example 2")}
                  />
                )}
              </div>
              <div
                className="flex items-center justify-between text-xl bg-primaryBlue text-white px-6 py-3 rounded-md  hover:bg-blue-50 hover:text-primaryBlue hover:border-[1.5px] hover:border-primaryBlue first-letter:duration-150 ease-in cursor-pointer"
                onClick={() => setInput("Example 3")}
              >
                Example 3
                {copied === "Example 3" ? (
                  <GoVerified
                    size={20}
                    className="object-contain  transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                  />
                ) : (
                  <BiCopy
                    size={20}
                    className="object-contain transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                    onClick={() => handleCopy("Example 3")}
                  />
                )}
              </div>
              <div
                className="flex items-center justify-between text-xl bg-primaryBlue text-white px-6 py-3 rounded-md  hover:bg-blue-50 hover:text-primaryBlue hover:border-[1.5px] hover:border-primaryBlue first-letter:duration-150 ease-in cursor-pointer"
                onClick={() => setInput("Example 4")}
              >
                Example 4
                {copied === "Example 4" ? (
                  <GoVerified
                    size={20}
                    className="object-contain transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                  />
                ) : (
                  <BiCopy
                    size={20}
                    className="object-contain transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                    onClick={() => handleCopy("Example 4")}
                  />
                )}
              </div>
            </div>
          </>
        ) : (
          // Chat conversation
          <>
            <div
              ref={conversationContainerRef}
              className="w-full top-0 overflow-y-auto py-[100px]"
            >
              {conversation?.map((message, index) => (
                <div
                  key={index}
                  className={`${
                    message.isUser
                      ? "bg-primaryWhite text-primaryBlue p-3"
                      : "bg-blue-100 text-primaryBlue p-3 "
                  }`}
                >
                  {message.isUser ? (
                    <div className="flex justify-between items-center md:px-[200px]">
                      {session?.user.provider === "credentials" || !session ? (
                        <MdAccountCircle size={30} className="m-4 ml-6 my-2" />
                      ) : (
                        <Image
                          src={session?.user.image}
                          width={40}
                          height={40}
                          className="m-4 mx-5 rounded-full my-0"
                        />
                      )}
                      {copied === message.text ? (
                        <GoVerified
                          size={20}
                          className="object-contain mr-8 transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                        />
                      ) : (
                        <BiCopy
                          size={20}
                          className="object-contain mr-8 transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                          onClick={() => handleCopy(message.text)}
                        />
                      )}
                    </div>
                  ) : (
                    <div className="flex justify-between items-center md:px-[200px]">
                      <BiSolidBot size={30} className="m-4 ml-6 my-2" />
                      <div className="flex items-center">
                        <BsInfoCircle
                          size={20}
                          className="object-contain mr-3 transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                          onClick={()=>handleToggleSourceOpen(message.source)}
                        />
                        <AiOutlineLike
                          size={20}
                          className="object-contain mr-3 transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                          onClick={() =>
                            handleOpenModal(
                              conversation[index - 1],
                              conversation[index],
                              "like"
                            )
                          }
                        />
                        <AiOutlineDislike
                          size={20}
                          className="object-contain mr-3 transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                          onClick={() =>
                            handleOpenModal(
                              conversation[index - 1],
                              conversation[index],
                              "dislike"
                            )
                          }
                        />
                        {copied === message.text ? (
                          <GoVerified
                            size={20}
                            className="object-contain mr-8 transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                          />
                        ) : (
                          <BiCopy
                            size={20}
                            className="object-contain mr-8 transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                            onClick={() => handleCopy(message.text)}
                          />
                        )}
                      </div>
                    </div>
                  )}
                  <div className="break-words md:px-[230px] py-2">
                    {message.isUser ? `${message.text}` : `${message.text}`}
                  </div>
                  <div className="flex justify-end mr-8 md:px-[200px]">
                    {message.time}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
        {/* Input Textbox */}
        <form
          className="w-full flex flex-col justify-center items-center fixed bottom-0 p-7 bg-slate-100"
          onSubmit={handleSubmit}
          style={{
            boxShadow:
              "rgba(14, 30, 37, 0.12) 0px 2px 4px 0px, rgba(14, 30, 37, 0.32) 0px 2px 16px 0px",
          }}
        >
          <div className="flex items-center w-full justify-center">
            {session?.user ? (
              <BsInfoCircle
                size={30}
                color="primaryBlue"
                className="object-contain mr-3 transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                onClick={handleToggleSourceClose}
              />
            ) : (
              <div className="flex items-center justify-center gap-3">
                {
                  <h1 className="text-xl text-primaryBlue font-bold">
                    {count}/20
                  </h1>
                }
                <BsInfoCircle
                  size={30}
                  color="primaryBlue"
                  className="object-contain mr-3 transition-opacity opacity-40 hover:opacity-100 hover:cursor-pointer"
                  onClick={handleToggleSourceClose}
                />
              </div>
            )}
            <input
              type="text"
              placeholder="Type your message..."
              className="w-10/12 px-4 py-2 rounded-lg border border-primaryBlue"
              onChange={(e) => {
                setInput(e.target.value);
              }}
              value={input}
              required
            />
            <button className="bg-primaryBlue text-white p-3 rounded-lg ml-2">
              <IoIosSend size={20} />
            </button>
            {isWaitingResponse && (
              <span className="typing-animation">
                <span className="dot-1"></span>
                <span className="dot-2"></span>
                <span className="dot-3"></span>
              </span>
            )}
          </div>
          {sourceOpen && (
            <Link
              className="flex sm:flex-row flex-col items-center justify-center cursor-pointer"
              href={
                source === undefined ||
                source === null ||
                source === "No Source Provided."
                  ? "#"
                  : source
              }
              target="blank"
            >
              <textarea
                name="source"
                id="source"
                cols="185"
                rows="3"
                className="settings_input m-5 relative sm:left-6 cursor-pointer"
                style={{ resize: "none" }}
                value={source}
              ></textarea>
            </Link>
          )}
        </form>
      </div>
    </div>
  );
};

export default Chatroom;
