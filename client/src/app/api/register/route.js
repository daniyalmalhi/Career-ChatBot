import { connectToDatabase } from "@/utils/database";
import User from "@/models/user";
import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";

export async function POST(req){
    try {
        const {email, password} = await req.json();
        const hashPassword = await bcrypt.hash(password, 10);
        await connectToDatabase();
        const date = new Date();
        await User.create({email, password: hashPassword, provider: "credentials", type: "Registered", date});

        return NextResponse.json({message: "User registered successfully"}, {status: 201})
    } catch (error) {
        console.log(error)
        return NextResponse.json({message: error.message}, {status: 500})
    }
}