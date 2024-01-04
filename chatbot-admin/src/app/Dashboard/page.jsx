"use client";
import { useSession, signOut } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import Spinner from "@/components/Spinner";
import { BsTrash } from "react-icons/bs";
import { format } from "date-fns";
import {toast} from 'react-toastify';

const Dashboard = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [defaultTypes, setDefaultTypes] = useState({});
  console.log(defaultTypes)
  const router = useRouter();
  const { data: session } = useSession();

  useEffect(() => {
    const fetchUsers = async () => {
      setLoading(true);
      const response = await fetch("/api/users");
      const data = await response.json();
      setUsers(data.users);
      setLoading(false);
    };
    fetchUsers();
  }, []);

  const handleTypeChange = (userId, selectedType) => {
    const updatedDefaultTypes = { ...defaultTypes, [userId]: selectedType };
    setDefaultTypes(updatedDefaultTypes);
  };

  const handleEdit = async (userId) => {
    if(!defaultTypes[userId]){
      toast.info('The type of user is not changed')
      return ;
    }
    // You can perform your edit logic here and console.log for testing.
    console.log(`Editing user ${userId} with type ${defaultTypes[userId]}`);
    try {
      const response = await fetch(`/api/users/${userId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({type: defaultTypes[userId]})
      })
      if(response.ok){
        window.location.reload();
      }
    } catch (error) {
      console.log(error)
      toast.error('Something went wrong')
    }
  };


  const handleDelete =  async (ID) =>{
    const response = await fetch(`/api/users/${ID}`, {
      method: 'DELETE',
    })
    if(response.status === 200){
      window.location.reload();
    }
  }

  return (
    <section className="flex flex-col items-center justify-center py-4 font-poppins mt-60 w-1/2">
      {loading ? (
        <div className="flex items-center justify-center my-10 ">
          <Spinner />
        </div>
      ) : (
        <div className="w-full my-5">
          {users
            .filter((user) => user.email !== "admin")
            .map((user, index) => (
              <div
                key={user._id}
                className="my-3 py-5 bg-slate-100 text-2xl text-primaryBlue rounded-md shadow-md flex items-center justify-between px-10 hover:shadow-lg"
              >
                <div className="w-full">
                  <p className="relative right-7 bottom-3 text-[18px] text-gray-500">
                    {index + 1}
                  </p>
                  <h1 className="font-bold">Email: {user.email}</h1>
                  <h1 className="text-md text-green-600">
                    Signed In with: <span className="font-bold">{user.provider}</span>
                  </h1>
                    <div className="mb-3 flex flex-col items-start">
                      <label htmlFor={`type-${user._id}`} className="text-md mr-5">
                        Type
                      </label>
                      <select
                        name={`type-${user._id}`}
                        id={`type-${user._id}`}
                        className="rounded-md px-10 text-[20px]"
                        value={defaultTypes[user._id] || user.type}
                        onChange={(e) => handleTypeChange(user._id, e.target.value)}
                      >
                        <option value="Registered">Registered</option>
                        <option value="Professional">Professional</option>
                      </select>
                    </div>
                    <button
                      className="py-1 px-5 text-sm bg-green-100 border-2 border-green-700 text-green-700 hover:bg-green-200 rounded-md cursor-pointer"
                      onClick={()=>handleEdit(user._id)}
                    >
                      Edit
                    </button>
                  <h1 className="text-sm text-right relative left-5 text-gray-500">
                    User Created: <span className="font-bold">{format(new Date(user.date), 'dd/MM/yyyy')}</span>
                  </h1>
                </div>
                <BsTrash color="darkred" className="cursor-pointer" onClick={()=>handleDelete(user._id)}/>
              </div>
            ))}
        </div>
      )}
    </section>
  );
};

export default Dashboard;
