import { connectToDatabase } from "@/utils/database";
import Feedback from "@/models/feedback";
import { NextResponse } from "next/server";

export async function POST(req) {
  try {
    await connectToDatabase();
    const { user, prompt, response, source, comments, thumbsUp } =
      await req.json();

    const newfeedback = new Feedback({
      user,
      prompt,
      response,
      source,
      comments,
      like_dislike: thumbsUp,
    });

    await newfeedback.save();

    return new NextResponse(
      { message: "Feedback Submitted successfully" },
      {
        status: 201,
      }
    );
  } catch (error) {
    console.log(error);
    return new NextResponse(
      { message: "Umm.. Something went wrong" },
      { status: 500 }
    );
  }
}
