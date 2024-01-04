import { Schema, model, models } from "mongoose";

const FeedbackSchema = new Schema(
  {
    user: {
      type: Schema.Types.ObjectId,
      ref: "User",
    },
    prompt: {
      type: String,
    },
    response: {
      type: String,
    },
    source: {
      type: String,
    },
    comments: {
      type: String,
    },
    like_dislike: {
      type: String,
    },
  },
  { timestamps: true }
);

const Feedback = models.Feedback || model("Feedback", FeedbackSchema);
export default Feedback;
