const mongoose = require("mongoose");

const conversationSchema = mongoose.Schema({
  message: {
    type: String,
  },
  room: {
    type: String,
  },
  isMe: {
    type: String,
  },
});

const convoSchema = mongoose.Schema({
  roomID: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "ChatRoom",
  },
  conversation: [conversationSchema],
});

module.exports = mongoose.model("Convo", convoSchema);
