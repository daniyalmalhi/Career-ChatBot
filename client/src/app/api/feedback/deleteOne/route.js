import { connectToDatabase } from "@/utils/database";
import Feedback from "@/models/feedback";

export async function DELETE(req) {
  try {
    await connectToDatabase();
    const { id } = await req.json();
    await Feedback.deleteOne({ _id: id });
    return new Response(
      JSON.stringify('Feedback Deleted Successfully'),
      { status: 200 }
    );
  } catch (error) {
    console.log(error);
    return new Response(
      JSON.stringify('Umm... Something went wrong!'),
      { status: 500 }
    );
  }
}
