import { connectToDatabase } from "@/utils/database";
import Feedback from "@/models/feedback";

export async function PUT(req) {
  try {
    await connectToDatabase();
    const { id, response, source, comments } = await req.json();
    const feedback = await Feedback.findOne({ _id: id });
    if (!feedback) {
      return new Response(JSON.stringify("Feedback not found"), {
        status: 404,
      });
    } else {
      feedback.response = response;
      feedback.source = source;
      feedback.comments = comments;
      feedback.like_dislike = 'like';
      await feedback.save();
      return new Response(JSON.stringify("Feedback updated successfully"), {
        status: 200,
      });
    }
  } catch (error) {
    console.log(error);
    return new Response(JSON.stringify("Umm.. Something went wrong"), {
      status: 500,
    });
  }
}
