import { connectToDatabase } from "@/utils/database";
import Conversation from "@/models/conversation";

export const dynamic = 'force-dynamic';
export const GET = async (req, {params}) => {
    try {
        await connectToDatabase();
        const convo = await Conversation.find({user: params.id});
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


export async function DELETE(req, {params}) {
    try {
      await connectToDatabase();
      try {
          await connectToDatabase();
          const convo = await Conversation.find({_id:params.id})
  
          if(convo.length == 0)
          {
              return new Response('Conversation not found', { status: 200 })
          }
          else
          {
              await Conversation.deleteOne({_id:params.id})
              return new Response('Conversation deleted', { status: 200 })
          }
      } catch (error) {
          
      }
      
    } catch (error) {
      console.error(error);
      return NextResponse.json({ message: error.message }, { status: 500 });
    }
  }
  