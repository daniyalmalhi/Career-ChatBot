import { useState, useEffect } from "react";
import io from "socket.io-client";
import { IoIosSend } from "react-icons/io";
import { HiOutlineLightBulb } from "react-icons/hi";
import { FaUserCircle } from "react-icons/fa";
import Navbar from "./components/Nav";
import Spinner from "./components/Spinner";

const socket = io.connect("http://localhost:3001");

//64f9babce2e79ce34f433d49 - Registered
//64f9f5067c7ebdde6a356668 - Registered
//64f9ac91e2e79ce34f433d45 - Professional

function App() {
  const [message, setMessage] = useState("");
  const [room, setRoom] = useState("");
  const [conversation, setConversation] = useState([]);
  const [type, setType] = useState("");
  const [users, setUsers] = useState([]);
  const [regUsers, setRegUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState("");
  //generate a random integer
  const queryParams = new URLSearchParams(window.location.search);
  const id = queryParams.get("id");
  // console.log("term",id)
  // const id = "64f9ac91e2e79ce34f433d45"; //id got from the url

  useEffect(() => {
    const checkUser = async () => {
      const response = await fetch("/api/users/check", {
        method: "POST",
        body: JSON.stringify({ id }),
        headers: { "Content-Type": "application/json" },
      });

      const json = await response.json();
      setType(json.type);
    };

    checkUser();
  }, [id]);

  useEffect(() => {
    const getUsers = async () => {
      const fetchUsers = await fetch("/api/users", {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      const users = await fetchUsers.json();
      setUsers(users.users);
    };
    getUsers();
  }, [id]);

  useEffect(() => {
    const getRegUsers = async () => {
      const fetchRegUsers = await fetch("/api/users/getRegUsers", {
        method: "POST",
        body: JSON.stringify({ id }),
        headers: { "Content-Type": "application/json" },
      });

      const regUsers = await fetchRegUsers.json();
      setRegUsers(regUsers.users);
    };
    getRegUsers();
  });

  useEffect(() => {
    //generate conversation
    const generateConvo = async () => {
      setLoading(true);
      const response = await fetch("/api/users/getConvo", {
        method: "POST",
        body: JSON.stringify({ roomID: room }),
        headers: { "Content-Type": "application/json" },
      });
      const json = await response.json();
      if (json.convo) {
        setConversation(json.convo.conversation);
        setLoading(false);
      } else {
        setConversation([]);
        setLoading(false);
      }
    };
    generateConvo();
  }, [room]);

  const joinRoom = (receiverID) => {
    //generate room number
    const generateRoom = async () => {
      setLoading(true);
      const response = await fetch("/api/users/room", {
        method: "POST",
        body: JSON.stringify({ sender: id, receiver: receiverID }),
        headers: { "Content-Type": "application/json" },
      });
      const json = await response.json();
      setRoom(json.room);
      socket.emit("join_room", json.room);
      setSelected(receiverID);
      setLoading(false);
    };

    generateRoom();
  };

  const sendMessage = (e) => {
    //send to DB
    e.preventDefault();
    const updateConvo = async () => {
      const response = await fetch("/api/users/convo", {
        method: "POST",
        body: JSON.stringify({ message, room, isMe: id }),
        headers: { "Content-Type": "application/json" },
      });
      const json = await response.json();
      console.log(json);
    };
    updateConvo();
    socket.emit("send_message", { message, room, isMe: id }); // Pass room with the message
    setConversation((prev) => [...prev, { message, room, isMe: id }]); // Add sent message with isMe: true
    setMessage(""); // Clear the input field after sending
  };

  useEffect(() => {
    const receiveMessage = (data) => {
      setConversation((prev) => [
        ...prev,
        { message: data.message, room: data.room, isMe: data.isMe },
      ]);
    };

    socket.on("receive_message", receiveMessage);

    // Cleanup the event listener when the component unmounts
    return () => {
      socket.off("receive_message", receiveMessage);
    };
  }, []);

  return (
    <div className="flex h-screen w-full">
      <Navbar />
      {/* Left panel for user names */}
      <div className="sm:w-1/4 w-1/2 bg-blue-100 overflow-y-auto sm:mt-18 mt-16">
        {/* Repeat this block for each user */}
        {users && users.length === 0 ? (
          <div className="pt-20 text-xl text-center">
            <p>No Users Added Yet!</p>
          </div>
        ) : type === "Registered" ? (
          users
            .filter(
              (user) => user.email !== "admin" && user.type === "Professional"
            )
            .map((user) => (
              <div
                className={`py-6 shadow-md cursor-pointer hover:shadow-lg mb-3 flex md:flex-row flex-col md:items-center w-full ${
                  selected === user._id
                    ? "bg-primaryBlue text-white"
                    : "bg-blue-100 text-black"
                }`}
                onClick={() => joinRoom(user._id)}
              >
                {}
                <FaUserCircle size={35} className=" md:flex hidden ml-2" />
                <div>
                  <h1 className="sm:text-xl text-sm md:ml-12 ml-2 truncate">
                    {user.email}
                  </h1>
                </div>
              </div>
            ))
        ) : regUsers && regUsers.length === 0 ? (
          <div className="pt-20 text-xl text-center">
            <p>No Users Added Yet!</p>
          </div>
        ) : (
          <div>
            {regUsers.map((user) => (
              <div
                className={`py-6 shadow-md cursor-pointer hover:shadow-lg mb-3 flex md:flex-row flex-col md:items-center w-full ${
                  selected === user._id
                    ? "bg-primaryBlue text-white"
                    : "bg-blue-100 text-black"
                }`}
                onClick={() => joinRoom(user._id)}
              >
                {}
                <FaUserCircle size={35} className=" md:flex hidden ml-2" />
                <div>
                  <h1 className="sm:text-xl text-sm md:ml-12 ml-2 truncate">
                    {user.email}
                  </h1>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Repeat block end */}
      </div>

      {/* Right panel for chat */}
      <div className="w-3/4 bg-white overflow-y-auto sm:mt-16 mt-16">
        {/* Center content vertically and horizontally */}
        {loading ? (
          <div className="flex items-center justify-center h-screen my-auto">
            <Spinner />
          </div>
        ) : conversation.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full">
            <h1 className="sm:text-6xl text-3xl mb-5 leading-none font-light text-primaryBlue text-center">
              Welcome to the Consulation Portal
            </h1>
            <h1 className="sm:text-3xl text-lg text-primaryBlue mb-5 text-center">
              Here you can have a conversation with professional users
            </h1>
            <div className="flex sm:flex-row flex-col items-center justify-center gap-2 sm:text-2xl text-md mb-5 px-1 text-center">
              <span>
                <HiOutlineLightBulb />
              </span>
              Type to start a conversation
            </div>
          </div>
        ) : (
          <div className="bg-white overflow-y-auto h-full">
            {conversation.map((message, index) => (
              <div
                key={index}
                className={`${
                  message.isMe === id
                    ? "bg-blue-100 text-primaryBlue p-5 text-xl"
                    : "bg-[#fcf2f2] text-primaryBlue p-5 text-xl "
                }`}
              >
                <div className="break-words md:px-[230px] py-2">
                  {message.isMe === id ? (
                    <div className="flex items-center gap-2">
                      <FaUserCircle className="text-primaryBlue" />
                      <h1>{message.message}</h1>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <FaUserCircle className="text-orange-700" />
                      <h1>{message.message}</h1>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="w-full flex flex-col justify-center items-center bottom-0 sm:p-7 py-5 px-2 bg-slate-100">
          <form
            className="flex items-center w-full justify-center"
            onSubmit={sendMessage}
          >
            <input
              type="text"
              placeholder="Type your message..."
              className="w-10/12 px-4 py-2 rounded-lg border border-primaryBlue"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              required
            />
            <button className="bg-primaryBlue text-white p-3 rounded-lg ml-2">
              <IoIosSend size={20} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
export default App;
