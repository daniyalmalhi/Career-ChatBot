import { connectToDatabase } from "@/utils/database";
import User from "@/models/user";
import { NextResponse } from "next/server";

export async function POST(req){
    try {
        await connectToDatabase();
        const {email} = await req.json();
        const user = await User.findOne({email}).select("_id");
        console.log("User: ", user)
        return NextResponse.json({user})
    } catch (error) {
        console.log(error)
        return NextResponse.json({message: error.message}, {status: 500})
    }
}