import { connectToDatabase } from "@/utils/database";
import User from "@/models/user";

export async function DELETE(req, {params}) {
    try {
      await connectToDatabase();
      try {
          await connectToDatabase();
          const user = await User.find({_id:params.id})
  
          if(user.length == 0)
          {
              return new Response('User not found', { status: 200 })
          }
          else
          {
              await User.deleteOne({_id:params.id})
              return new Response('User deleted', { status: 200 })
          }
      } catch (error) {
          
      }
      
    } catch (error) {
      console.error(error);
      return NextResponse.json({ message: error.message }, { status: 500 });
    }
  }


  export async function PUT(req, { params }) {
    try {
      const { type } = await req.json();
      await connectToDatabase();
      try {
        const user = await User.findOne({ _id: params.id }); // Use findOne instead of find
        console.log(user);
  
        if (!user) {
          return new Response('User not found', { status: 404 }); // Return 404 status for not found
        } else {
          user.type = type; // Update the user's type directly
          await user.save(); // Save the updated user
          return new Response('User Updated', { status: 200 });
        }
      } catch (error) {
        // Handle the error properly
        console.error(error);
        return new Response({ message: error.message }, { status: 500 });
      }
    } catch (error) {
      // Handle the error properly
      console.error(error);
      return new Response({ message: error.message }, { status: 500 });
    }
  }