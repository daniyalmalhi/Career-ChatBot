const mongoose = require("mongoose");

const chatroomSchema = mongoose.Schema({
  sender: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
  },
  receiver: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
  },
  room: {
    type: String
  },
});

module.exports = mongoose.model("ChatRoom", chatroomSchema);
