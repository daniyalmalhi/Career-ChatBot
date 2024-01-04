import { connectToDatabase } from "@/utils/database";
import User from "@/models/user";
import { NextResponse } from "next/server";


export const dynamic = 'force-dynamic';
export async function GET(req){
    try {
        await connectToDatabase();
        const users = await User.find({});
        if(!users){
            return NextResponse.json({message: 'No users found'}, {status: 404})
        }
        return NextResponse.json({users}, {status: 200})
    } catch (error) {
        console.log(error)
        return NextResponse.json({message: error.message}, {status: 500})
    }
}