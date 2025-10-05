const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String },          // only for manual signup
  googleId: { type: String }           // only for Google signup
}, { timestamps: true });

module.exports = mongoose.model("User", userSchema);
