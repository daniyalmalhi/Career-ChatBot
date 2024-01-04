import { Schema, model, models } from "mongoose";

const convoSchema = new Schema({
  isUser: {
    type: Boolean,
  },
  text: {
    type: String,
  },
  source: {
    type: String,
  },
  time: {
    type: String,
  },
});

const contextSchema = new Schema({
  role: {
    type: String,
  },
  content: {
    type: String,
  },
});

const ConversationSchema = new Schema(
  {
    user: {
      type: Schema.Types.ObjectId,
      ref: "User",
    },
    count: {
      type: Number,
    },
    title: {
      type: String,
    },
    date: {
      type: Date,
    },
    conversation: [convoSchema],
    context: [contextSchema],
    feedback: {
      type: String,
    },
    video_transcript: {
      type: String,
    },
    prompts: {
      type: String,
    },
  },
  { timestamps: true }
);

const Conversation =
  models.Conversation || model("Conversation", ConversationSchema);
export default Conversation;
