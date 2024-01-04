import { connectToDatabase } from "@/utils/database";
import Conversation from "@/models/conversation";

export const dynamic = 'force-dynamic';

export const GET = async (req, {params}) => {
    try {
        await connectToDatabase();
        const convo = await Conversation.find({_id: params.id});
        if (!convo) {
            return new Response("No Conversation yet created", { status: 400 });
        } else {
            const response = new Response(JSON.stringify(convo), { status: 200 });
            return response;
        }
    } catch (error) {
        console.log(error);
        return new Response("Failed to fetch convo", { status: 500 });
    }
};