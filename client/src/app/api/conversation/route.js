import { connectToDatabase } from "@/utils/database";
import Conversation from "@/models/conversation";
import User from "@/models/user";
import { NextResponse } from "next/server";

export async function POST(req) {
  try {
    await connectToDatabase();
    const data = await req.json();
    const { conversation, context, count, user } = data;
    let date = new Date();

    if (user) {
      // Generate a unique title for the conversation
      const title = conversation[0].text;

      try {
        // Check if a conversation with the generated title already exists
        const existingConversation = await Conversation.findOne({
          title,
        }).select("_id");
        const existingUser = await User.findOne({ email: user }).select("_id");
        const userId = existingUser.id;

        if (existingConversation) {
          // If it exists, append the new conversation to the existing one
          existingConversation.conversation = conversation;
          existingConversation.context = context;
          existingConversation.count = count;
          existingConversation.date = date;
          await existingConversation.save();
          return new Response("Conversation updated", { status: 200 });
        } else {
          // If it doesn't exist, create a new conversation
          await Conversation.create({
            user: userId,
            count,
            title,
            conversation,
            context,
            date,
          });

          return new Response("Conversation created", { status: 200 });
        }
      } catch (error) {
        console.error("Error handling conversation:", error);
      }
    } else {
      return new Response("User not registered", { status: 200 });
    }
  } catch (error) {
    console.error(error);
    return NextResponse.json({ message: error.message }, { status: 500 });
  }
}
