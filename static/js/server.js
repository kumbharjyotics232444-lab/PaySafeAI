
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const bodyParser = require("body-parser");
const bcrypt = require("bcrypt");

const app = express();
app.use(cors());
app.use(bodyParser.json());

mongoose.connect("mongodb+srv://kumbharjyotics232444:2rJuzrAMbUS9ZtQB@cluster0.rlkth.mongodb.net/PaySafeAI?retryWrites=true&w=majority", {
  dbName: "PaySafeAI"
})
.then(() => console.log("✅ MongoDB Connected"))
.catch(err => console.error("❌ DB Connection Error:", err));


const userSchema = new mongoose.Schema({
  name: String,
  email: { type: String, unique: true },
  password: String
});

const User = mongoose.models.User || mongoose.model("users", userSchema, "users");

// Signup
app.post("/signup", async (req, res) => {
  try {
    const { name, email, password } = req.body;
    const existingUser = await User.findOne({ email });
    if (existingUser) return res.status(400).json({ message: "User already exists" });

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ name, email, password: hashedPassword });
    await newUser.save();

    res.status(201).json({ message: "User registered successfully" });
  } catch (err) {
    res.status(500).json({ message: "Error creating user", error: err.message });
  }
});

// Login
app.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    // Simple admin check
    if (email === "admin@gmail.com" && password === "admin123") {
      return res.status(200).json({ message: "Admin login successful", role: "admin" });
    }

    const user = await User.findOne({ email });
    if (!user) return res.status(400).json({ message: "Invalid email or password" });

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ message: "Invalid email or password" });

    res.status(200).json({
      message: "User login successful",
      role: "user",
      user: { name: user.name, email: user.email }
    });
  } catch (err) {
    res.status(500).json({ message: "Error logging in", error: err.message });
  }
});

// Get all users
app.get("/admin/users", async (req, res) => {
  try {
    const users = await User.find({}, { password: 0 }); // exclude password
    res.json(users);
  } catch (err) {
    res.status(500).json({ message: "Error fetching users", error: err.message });
  }
});

// Delete user
app.delete("/admin/users/:id", async (req, res) => {
  try {
    const { id } = req.params;
    await User.findByIdAndDelete(id);
    res.json({ message: "User deleted successfully" });
  } catch (err) {
    res.status(500).json({ message: "Error deleting user", error: err.message });
  }
});

// Get user by email
app.get("/user/:email", async (req, res) => {
  try {
    const email = req.params.email;
    const user = await User.findOne({ email }, { password: 0 }); // exclude password
    if (!user) return res.status(404).json({ message: "User not found" });
    res.json(user);
  } catch (err) {
    res.status(500).json({ message: "Error fetching user", error: err.message });
  }
});

// Update user
app.put("/user/:email", async (req, res) => {
  try {
    const { name, email, password } = req.body;
    const updateData = { name, email };
    if (password) updateData.password = await bcrypt.hash(password, 10);

    const user = await User.findOneAndUpdate({ email: req.params.email }, updateData, { new: true });
    if (!user) return res.status(404).json({ message: "User not found" });

    res.json({ message: "Profile updated successfully", user });
  } catch (err) {
    res.status(500).json({ message: "Error updating profile", error: err.message });
  }
});

// History Table Schema
const historySchema = new mongoose.Schema({
  intended_balcon_amount: Number,
  customer_age: Number,
  income: Number,
  payment_type: String,
  device_os: String,
  session_length_in_minutes: Number,
  credit_risk_score: Number,
  prediction: String,
  createdAt: { type: Date, default: Date.now }
});

const History = mongoose.models.History || mongoose.model("history", historySchema, "history");

const { PythonShell } = require("python-shell");

// Predict Route
app.post("/predict", async (req, res) => {
  try {
    const data = req.body;

    // Call Python script
    let options = {
      mode: "text",
      pythonOptions: ["-u"],
      args: [JSON.stringify(data)],
    };

    PythonShell.run("predict.py", options, async (err, results) => {
      if (err) {
        console.error("Python Error:", err);
        return res.status(500).json({ error: "Model prediction failed" });
      }

      const prediction = results[0];

      // Save to MongoDB
      const newEntry = new History({
        ...data,
        prediction,
      });
      await newEntry.save();

      res.json({ prediction });
    });
  } catch (err) {
    console.error("Predict Error:", err);
    res.status(500).json({ error: "Server error" });
  }
});


// Get History
app.get("/history", async (req, res) => {
  try {
    const history = await History.find().sort({ createdAt: -1 });
    res.json(history);
  } catch (err) {
    res.status(500).json({ message: "Error fetching history", error: err.message });
  }
});

app.listen(5000, () => {
  console.log("🚀 Server running on http://localhost:5000");
});
