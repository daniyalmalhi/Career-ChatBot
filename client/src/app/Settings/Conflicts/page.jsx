"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation"; // Import the router
import Spinner from "@/components/Spinner";
import Image from "next/image";
import { toast } from "react-toastify";

const Conflicts = () => {
  const router = useRouter(); // Initialize the router

  const [feedback, setFeedback] = useState([]);
  const [loading, setLoading] = useState(true);
  const [editStates, setEditStates] = useState([]);
  const [deletedFeedback, setDeletedFeedback] = useState([]);
  const [editedFeedback, setEditedFeedback] = useState([]);
  const [deleteAllLoader, setDeleteAllLoader] = useState(false);
  const [saveLoader, setSaveLoader] = useState(false);
  const [sendAllLoader, setSendAllLoader] = useState(false);
  const [selectAll, setSelectAll] = useState(false);
  const [filter, setFilter] = useState("all");

  const handleSelectAllModal = () => {
    setSendAllLoader(true);
    let selectedFeedback = [];
    feedback.map((item) => {
      if (deletedFeedback.includes(item._id)) {
        selectedFeedback.push(item.prompt, item.response, item.source);
      }
    });
    const sendFeedback = async () => {
      const response = await fetch("http://localhost:8080/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(selectedFeedback), // Send the user's input as an object
      });
      const data = await response.json();
      setSendAllLoader(false);
      if (data === "Feedback Trained Successfully") {
        await fetch("/api/feedback/deleteMany", {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ ids: deletedFeedback }),
        });
        setDeletedFeedback([]);
        setSelectAll(false);
        window.location.reload();
        toast.success(data);
      } else toast.error("Something went wrong");
    };
    sendFeedback();
  };

  const handleEditFeedback = (id, response, source, comments) => {
    setSaveLoader(true);
    const editFeedback = async () => {
      const res = await fetch("/api/feedback/edit", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ id, response, source, comments }),
      });
      const data = await res.json();
      const status = res.status;
      setSaveLoader(false);
      if (status === 200) {
        toast.success(data);
        window.location.reload();
      } else {
        toast.error(data);
      }
    };
    editFeedback();
  };

  const handleDeleteAll = () => {
    setDeleteAllLoader(true);
    const deleteFeedbacks = async () => {
      const res = await fetch("/api/feedback/deleteMany", {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ids: deletedFeedback }),
      });
      const data = await res.json();
      const status = res.status;
      setDeleteAllLoader(false);
      if (status === 200) {
        toast.success(data);
        window.location.reload();
      } else {
        toast.error(data);
      }
    };
    deleteFeedbacks();
  };

  useEffect(() => {
    const fetchFeedback = async () => {
      const response = await fetch("/api/feedback/get");
      const data = await response.json();
      setFeedback(data);
      setLoading(false);
    };
    fetchFeedback();
  }, []);

  const handleShowEdit = (index) => {
    setEditStates((prevEditStates) => {
      const newEditStates = [...prevEditStates];
      newEditStates[index] = !newEditStates[index];
      return newEditStates;
    });
  };

  const handleToggleSelectAll = () => {
    if (selectAll) {
      // If "Select All" is currently active, deselect all feedback items
      setDeletedFeedback([]);
    } else {
      // If "Select All" is not active, select all feedback items
      const allFeedbackIds = feedback.map((item) => item._id);
      setDeletedFeedback(allFeedbackIds);
    }
    setSelectAll(!selectAll); // Toggle the "Select All" state
  };

  const handleSelect = (index) => {
    setDeletedFeedback((prevDeletedFeedback) => {
      if (prevDeletedFeedback.includes(index)) {
        // If the index is already present, remove it
        return prevDeletedFeedback.filter((item) => item !== index);
      } else {
        // If the index is not present, add it
        return [...prevDeletedFeedback, index];
      }
    });
  };

  const handleFilter = (filterType) => {
    if (filter === filterType) {
      // If the same filter is clicked again, toggle it off
      setFilter("all");
    } else {
      // Otherwise, set the filter to the selected filter
      setFilter(filterType);
    }
  };

  if (loading)
    return (
      <div className="flex items-center justify-center h-screen">
        <Spinner />;
      </div>
    );

  if (!feedback.length)
    return (
      <div className="flex items-center justify-center h-screen">
        <h1 className="text-5xl font-bold">No Feedback Yet :(</h1>
      </div>
    );

  const filteredFeedback = feedback.filter((item) => {
    if (filter === "positive" && item.like_dislike === "like") {
      return true;
    } else if (filter === "negative" && item.like_dislike === "dislike") {
      return true;
    }
    return filter === "all";
  });

  return (
    <section className="bg-slate-100 h-screen sm:p-10 p-5 sm:m-10 m-5 w-full">
      <section>
        <div className="flex items-center my-10 mt-20 justify-center ">
          <Image
            src="/assets/back-button.png"
            width={30}
            height={30}
            className="object-contain mr-5 hover:cursor-pointer"
            alt="gobackbtn"
            onClick={() => router.push("/Settings")} // Use the router to navigate
          />
          <h1 className="text-4xl">Feedback Pool</h1>
        </div>
        <div className="flex sm:flex-row flex-col items-center justify-center gap-4 mb-5">
          <button
            className={`border-4 border-green-600 font-poppins font-bold  duration-200 ease-linear p-2 rounded-sm w-full px-10 ${
              filter === "positive"
                ? "bg-green-600 text-white"
                : " text-green-600 bg-white hover:bg-green-100"
            }`}
            onClick={() => handleFilter("positive")}
          >
            Show Positive Feedback Only
          </button>
          <button
            className={`border-4 border-red-600  font-poppins font-bold  duration-200 ease-linear p-2 rounded-sm w-full px-10 ${
              filter === "negative"
                ? "bg-red-600 text-white"
                : "text-red-600 bg-white hover:bg-red-100"
            }`}
            onClick={() => handleFilter("negative")}
          >
            Show Negative Feedback Only
          </button>
        </div>
        <button
          className={`border-4 border-primaryBlue font-poppins font-bold  duration-200 ease-linear p-2 rounded-sm w-full px-10 mb-20 ${
            selectAll === "negative"
              ? "bg-primaryBlue text-white"
              : "text-primaryBlue bg-white hover:bg-blue-100"
          }`}
          onClick={handleToggleSelectAll}
        >
          {selectAll ? "Unselect All" : "Select All"}
        </button>
        <div>
          {filteredFeedback.map((item, index) => (
            <div
              className="flex sm:flex-row flex-col items-center"
              key={item._id}
            >
              <div
                className={`sm:w-1/2 flex flex-col justify-center sm:p-10 p-5 sm:my-5 my-3 text-xl shadow-xl rounded-lg bg-slate-50 gap-5 border-4 ${
                  item.like_dislike === "like"
                    ? "shadow-green-400 border-green-300"
                    : "shadow-red-400 border-red-300"
                }`}
              >
                <input
                  type="checkbox"
                  className="ml-auto w-6 h-6 text-primaryBlue bg-white border-gray-300 rounded-xl focus:ring-blue-500 focus:ring-2"
                  checked={deletedFeedback.includes(item._id)}
                  onChange={() => handleSelect(item._id)}
                />
                <label className="font-bold text-primaryBlue">Prompt</label>
                <input
                  type="text"
                  value={item.prompt}
                  readOnly
                  disabled
                  className="settings_input bg-white"
                />
                <label className="font-bold text-primaryBlue">Response</label>
                <textarea
                  cols="30"
                  rows="10"
                  value={item.response}
                  className="settings_input relative cursor-pointer"
                />
                <label className="font-bold text-primaryBlue">Source</label>
                <input
                  type="text"
                  value={item.source}
                  readOnly
                  disabled
                  className="settings_input bg-white"
                />
                <label className="font-bold text-primaryBlue">Comments</label>
                <input
                  type="text"
                  value={item.comments}
                  readOnly
                  disabled
                  className="settings_input bg-white"
                />
                <div>
                  <button
                    className="border-2 bg-primaryBlue text-white font-poppins hover:bg-slate-600 duration-200 ease-linear p-2 rounded-sm w-full px-10"
                    onClick={() => handleShowEdit(index)}
                  >
                    {editStates[index] ? "Close" : "Edit"}
                  </button>
                </div>
              </div>
              <div
                className={`${
                  editStates[index]
                    ? "sm:w-1/2 flex flex-col justify-center sm:p-10 p-5 sm:m-5 my-3 text-xl shadow-xl rounded-lg bg-slate-50 gap-5 border-4 shadow-orange-400 border-yellow-300"
                    : "hidden"
                }`}
              >
                <label className="font-bold text-primaryBlue">Prompt</label>
                <input
                  type="text"
                  value={item.prompt}
                  readOnly
                  disabled
                  className="settings_input bg-white"
                />
                <label className="font-bold text-primaryBlue">Response</label>
                <textarea
                  cols="30"
                  rows="12"
                  className="settings_input relative cursor-pointer"
                  value={editedFeedback[index]?.response || ""} // Use optional chaining to access editedFeedback
                  onChange={(e) => {
                    setEditedFeedback((prevEditedFeedback) => {
                      const newEditedFeedback = [...prevEditedFeedback];
                      if (!newEditedFeedback[index])
                        newEditedFeedback[index] = {}; // Initialize if not exist
                      newEditedFeedback[index].response = e.target.value;
                      return newEditedFeedback;
                    });
                  }}
                />
                <label className="font-bold text-primaryBlue">Source</label>
                <input
                  type="text"
                  className="settings_input bg-white"
                  value={editedFeedback[index]?.source || ""} // Use optional chaining
                  onChange={(e) => {
                    setEditedFeedback((prevEditedFeedback) => {
                      const newEditedFeedback = [...prevEditedFeedback];
                      if (!newEditedFeedback[index])
                        newEditedFeedback[index] = {}; // Initialize if not exist
                      newEditedFeedback[index].source = e.target.value;
                      return newEditedFeedback;
                    });
                  }}
                />
                <label className="font-bold text-primaryBlue">Comments</label>
                <input
                  type="text"
                  className="settings_input bg-white"
                  value={editedFeedback[index]?.comments || ""} // Use optional chaining
                  onChange={(e) => {
                    setEditedFeedback((prevEditedFeedback) => {
                      const newEditedFeedback = [...prevEditedFeedback];
                      if (!newEditedFeedback[index])
                        newEditedFeedback[index] = {}; // Initialize if not exist
                      newEditedFeedback[index].comments = e.target.value;
                      return newEditedFeedback;
                    });
                  }}
                />
                <div>
                  {saveLoader ? (
                    <button className="border-2 bg-primaryBlue text-white font-poppins hover:bg-slate-600 duration-200 ease-linear p-2 rounded-sm w-full px-10 flex items-center justify-center">
                      <Spinner />
                    </button>
                  ) : (
                    <button
                      className="border-2 bg-primaryBlue text-white font-poppins hover:bg-slate-600 duration-200 ease-linear p-2 rounded-sm w-full px-10"
                      onClick={() =>
                        handleEditFeedback(
                          item._id,
                          editedFeedback[index].response,
                          editedFeedback[index].source,
                          editedFeedback[index].comments
                        )
                      }
                    >
                      Save
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>
      <div
        className={`fixed inset-0 h-fit mt-auto ${
          deletedFeedback.length ? "flex items-end justify-center" : "hidden"
        }`}
      >
        <div className="bg-primaryBlue opacity-95 sm:p-10 p-5 gap-3 m-1 rounded-md flex sm:flex-row flex-col items-center justify-center text-white w-full">
          {deleteAllLoader ? (
            <button className="rounded-sm bg-slate-200 text-primaryBlue sm:text-lg text-md font-bold w-full px-10 py-2 flex items-center justify-center">
              <Spinner />
            </button>
          ) : (
            <button
              className="rounded-sm bg-slate-200 text-primaryBlue sm:text-lg text-md font-bold w-full px-10 py-2"
              onClick={handleDeleteAll}
            >
              {deletedFeedback.length > 1 ? "Delete All" : "Delete"}
            </button>
          )}
          {sendAllLoader ? (
            <button className="rounded-sm bg-slate-200 text-primaryBlue sm:text-lg text-md font-bold w-full px-10 py-2 flex items-center justify-center">
              <Spinner />
            </button>
          ) : (
            <button
              className="rounded-sm bg-slate-200 text-primaryBlue sm:text-lg text-md font-bold w-full px-10 py-2"
              onClick={handleSelectAllModal}
            >
              {deletedFeedback.length > 1
                ? "Send All To Train"
                : "Send To Train"}
            </button>
          )}
        </div>
      </div>
    </section>
  );
};

export default Conflicts;
