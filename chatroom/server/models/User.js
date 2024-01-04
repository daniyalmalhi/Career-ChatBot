const mongoose = require("mongoose");

const userSchema = mongoose.Schema({
  email: {
    type: String,
    unique: true,
    required: [true, "Please provide an email address"],
  },
  password: {
    type: String,
  },
  provider: {
    type: String,
  },
  username: {
    type: String,
    match: [
      /^(?=.{8,20}$)(?![_.])(?!.*[_.]{2})[a-zA-Z0-9._]+(?<![_.])$/,
      '"Username invalid, it should contain 8-20 alphanumeric letters and be unique!"',
    ],
  },
  image: {
    type: String,
  },
  type: {
    type: String,
  },
});

module.exports = mongoose.model("User", userSchema);
