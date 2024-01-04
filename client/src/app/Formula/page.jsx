"use client";
import { useState, useEffect } from "react";
import { BiMath } from "react-icons/bi";
import { IoIosSend } from "react-icons/io";
import { AiOutlineClose } from "react-icons/ai";
import Spinner from "@/components/Spinner";
import { toast } from "react-toastify";

const Formula = () => {
  const [formulas, setFormulas] = useState([]);
  const [addFormula, setAddFormula] = useState("");
  const [selectedFormula, setSelectedFormula] = useState(null);
  const [selectedParameters, setSelectedParameters] = useState([]);
  const [parameterValues, setParameterValues] = useState({});
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [addLoader, setAddLoader] = useState(false);
  const [formulaLoader, setFormulaLoader] = useState(true);

  useEffect(() => {
    const getFormulas = async () => {
      const res = await fetch("http://localhost:8080/getFormulas");
      const data = await res.json();
      setFormulas(data);
      setFormulaLoader(false);
    };
    getFormulas();
  }, []);

  const handleAddFormula = async () => {
    setAddLoader(true);
    const res = await fetch("http://localhost:8080/addFormula", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        formula: addFormula,
      }),
    });
    const data = await res.json();
    setAddLoader(false);
    if (data) {
      toast.success("Formula Added Successfully");
      setAddFormula("");
      window.location.reload();
    } else {
      toast.error("Error Occured");
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    const response = await fetch("http://localhost:8080/calculateFormula", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        formula: selectedFormula,
        parameters: parameterValues,
      }),
    });
    const data = await response.json();
    setLoading(false);
    setResult(data);
  };

  const handleFormulaSelect = (formulaObj) => {
    setSelectedFormula(formulaObj.formula);
    setSelectedParameters(formulaObj.parameters);
    setParameterValues(
      formulaObj.parameters.reduce((acc, param) => {
        acc[param] = ""; // Initialize all parameters with empty string
        return acc;
      }, {})
    );
  };

  const handleParameterChange = (param, value) => {
    setParameterValues((prev) => ({ ...prev, [param]: value }));
  };

  const handleCloseFormulaSelect = () => {
    setSelectedFormula(null);
    setSelectedParameters([]);
    setParameterValues({});
    setResult("");
  };

  return (
    <section className="bg-white p-10 flex flex-col items-center justify-center rounded-md">
      <div className="flex items-center w-full">
        <input
          type="text"
          value={addFormula}
          onChange={(e) => setAddFormula(e.target.value)}
          placeholder="Add Formula"
          className="p-2 rounded-md border-2 border-primaryBlue w-full my-5"
        />
        {addLoader ? (
          <div className="ml-2">
            <Spinner />
          </div>
        ) : (
          <button
            className="bg-primaryBlue text-white p-3 rounded-lg ml-2"
            onClick={handleAddFormula}
          >
            <IoIosSend size={20} />
          </button>
        )}
      </div>
      <h1 className="text-4xl mb-5 leading-none text-primaryBlue text-center">
        Extracted Formulas
      </h1>
      {formulaLoader ? <div className="flex items-center justify-center"><Spinner/></div> : formulas.length === 0 ? (
        <h2 className="text-2xl text-center">No formulas found</h2>
      ) : (
        <div className="text-lg my-5">
          {formulas.map((formulaObj, index) => (
            <div
              key={index}
              className="flex items-center justify-between px-5 py-2 my-2 shadow-md rounded-lg text-primaryBlue cursor-pointer"
              onClick={() => handleFormulaSelect(formulaObj)}
            >
              <h2>{formulaObj.formula}</h2>
              <button className="p-1 ml-5 rounded-md text-white bg-slate-900 hover:text-black hover:bg-white hover:border-2 hover:border-slate-900 duration-150 ease-linear">
                <BiMath />
              </button>
            </div>
          ))}
        </div>
      )}
      <div
        className={`fixed inset-0 z-50 ${
          selectedFormula ? "flex items-center justify-center" : "hidden"
        }`}
        style={{
          backgroundColor: "rgba(0, 0, 0, 0.4)", // Fallback background color for older browsers
          backdropFilter: selectedFormula ? "blur(10px)" : "none", // Apply blur when modal is open
        }}
      >
        <div className="bg-white p-10 rounded-md">
          <AiOutlineClose
            className="text-3xl cursor-pointer ml-auto"
            onClick={handleCloseFormulaSelect}
            color="black"
          />
          <div className="flex items-center gap-x-10">
            <div>
              <h3 className="text-4xl text-primaryBlue mb-5">
                Selected Formula
              </h3>
              <p className="text-2xl font-semibold mb-5">{selectedFormula}</p>
              <h4 className="text-4xl text-primaryBlue mb-3">Parameters</h4>
              {selectedParameters.map((param, index) => (
                <div className="flex flex-col items-start" key={index}>
                  <div className="flex items-center justify-center w-full">
                    <input
                      type="number"
                      placeholder={`Enter the value of ${param}`}
                      className="mb-2 p-2 border rounded-md w-full"
                      value={parameterValues[param] || ""}
                      onChange={(e) =>
                        handleParameterChange(param, e.target.value)
                      }
                    />
                  </div>
                </div>
              ))}
              {loading ? (
                <button className="bg-primaryBlue px-10 py-3 text-white rounded-md mt-5 w-full flex items-center justify-center">
                  <Spinner />
                </button>
              ) : (
                <button
                  className="bg-primaryBlue px-10 py-3 text-white rounded-md mt-5 hover:bg-white hover:text-primaryBlue hover:border-2 hover:border-primaryBlue w-full"
                  onClick={handleSubmit}
                >
                  Submit
                </button>
              )}
            </div>
            <div className={!result ? "hidden" : ""}>
              <h3 className="text-4xl text-primaryBlue mb-5 text-center">
                Result
              </h3>
              <textarea
                cols="60"
                rows="15"
                readOnly
                className="border-4 border-green-700 rounded-md p-10 text-xl bg-white"
                value={result}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Formula;
