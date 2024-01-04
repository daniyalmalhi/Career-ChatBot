import { connectToDatabase } from "@/utils/database";
import Feedback from "@/models/feedback";

export async function DELETE(req) {
  try {
    await connectToDatabase();
    const { ids } = await req.json();
    await Feedback.deleteMany({ _id: { $in: ids } });
    return new Response(
      JSON.stringify('Feedbacks Deleted Successfully'),
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
