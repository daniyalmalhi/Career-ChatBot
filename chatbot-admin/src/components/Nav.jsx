"use client";
import Link from "next/link";
const Navbar = () => {
  return (
    <div className="shadow-md fixed top-0 left-0 w-full z-10">
      <div className="bg-primaryBlue md:flex py-4 items-center justify-between text-white">
        <Link href={'/'} className="flex items-center justify-center w-full">
        <div className="text-3xl font-normal font-poppins px-4">
          Admin Panel
        </div>
        </Link>

      </div>
    </div>
  );
};

export default Navbar;
