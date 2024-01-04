import { connectToDatabase } from "@/utils/database";
import Feedback from "@/models/feedback";
import { NextResponse } from "next/server";


export const dynamic = 'force-dynamic';
export const GET = async (req) => {
try {
    await connectToDatabase();
    const feedback = await Feedback.find({});
    if(!feedback) {
        return new NextResponse(
            { message: "No feedback found" },
            { status: 404 }
          );
    }
    else{
        return new NextResponse(
            JSON.stringify(feedback) ,
            { status: 200 }
          );
    }

} catch (error) {
    console.log(error)
    return new NextResponse(
        { message: "Umm.. Something went wrong" },
        { status: 500 }
      );
}
}
