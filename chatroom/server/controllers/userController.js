const asyncHandler = require("express-async-handler");
const User = require("../models/User");
const ChatRoom = require("../models/ChatRoom");
const Convo = require("../models/Convo");
const mongoose = require("mongoose");

const checkUser = asyncHandler(async (req, res) => {
  const { id } = req.body;

  if (!id) {
    return res.status(400).json({
      message: "ID is required",
      valid: false,
      type: "Invalid",
    });
  }

  // Check if the ID is a valid MongoDB ObjectID
  if (!mongoose.Types.ObjectId.isValid(id)) {
    return res.status(400).json({
      message: "Invalid ID format",
      valid: false,
      type: "Invalid",
    });
  }

  const userExists = await User.findOne({ _id: id });

  if (!userExists) {
    return res.status(404).json({
      message: "User Not Found",
      valid: false,
      type: "Invalid",
    });
  } else {
    //check the type of the user
    const type = userExists.type;
    return res.status(200).json({
      message: "User Exists",
      valid: true,
      type: type,
    });
  }
});

const getUsers = asyncHandler(async (req, res) => {
  const users = await User.find({});
  if (!users) {
    return res.status(400).json({
      message: "No Users Found",
    });
  } else {
    return res.status(200).json({
      users,
    });
  }
});

const generateChatRoom = asyncHandler(async (req, res) => {
  const { sender, receiver } = req.body;

  const room = [sender, receiver].sort().join("-");

  const roomExists = await ChatRoom.findOne({ room: room });

  if (roomExists) {
    return res
      .status(200)
      .json({ message: "Chatroom already exists", room: room });
  } else {
    const chatRoom = await ChatRoom.create({
      sender: new mongoose.Types.ObjectId(sender),
      receiver: new mongoose.Types.ObjectId(receiver),
      room: room,
    });
    await chatRoom.save();
    return res.status(200).json({ message: "Chatroom created", room: room });
  }
});

const updateConvo = asyncHandler(async (req, res) => {
  try {
    const { message, room, isMe } = req.body;

    // Find the ChatRoom by room name
    const chatRoom = await ChatRoom.findOne({ room });

    if (!chatRoom) {
      return res.status(400).json({ message: "Room not found" });
    }

    // Find the Convo document by roomID
    const convo = await Convo.findOne({ roomID: chatRoom._id });

    if (!convo) {
      // Create a new convo if it doesn't exist
      const newConvo = new Convo({
        roomID: chatRoom._id,
        conversation: [{ message, room, isMe }],
      });
      await newConvo.save();
      return res.status(200).json({ message: "Conversation created" });
    } else {
      // Update the existing convo
      convo.conversation.push({ message, room, isMe });
      await convo.save();
      return res.status(200).json({ message: "Conversation updated" });
    }
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Internal server error" });
  }
});

const getRegUsers = asyncHandler(async (req, res) => {
  //getting ID of professional user
  const { id } = req.body;
  //fetch all registered users that are talking to this ID
  const regUsers = await ChatRoom.find({ receiver: id }).select("sender");
  // console.log(regUsers)
  //get all users from the user collection that matches with regUsers.sender
  const IDs = regUsers.map((user) => user.sender);
  const users = await User.find({ _id: { $in: IDs } });
  return res.status(200).json({ users });
});

const getConvo = asyncHandler(async (req, res) => {
  const { roomID } = req.body;
  const room = await ChatRoom.findOne({ room: roomID }).select("_id");
  console.log("room", room)
  const convo = await Convo.findOne({ roomID: room }).select("conversation");
  console.log("convo",convo)
  if (!convo) return res.status(400).json({ message: "No conversation found" });
  else return res.status(200).json({ convo });
});

module.exports = {
  checkUser,
  getUsers,
  generateChatRoom,
  updateConvo,
  getRegUsers,
  getConvo,
};
