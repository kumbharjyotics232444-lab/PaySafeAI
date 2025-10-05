require("dotenv").config();  // Loads env variables
const express = require("express");
const mongoose = require("mongoose");
const session = require("express-session");
const passport = require("passport");
const GoogleStrategy = require("passport-google-oauth20").Strategy;
const bcrypt = require("bcrypt");
const User = require("./User");  

const app = express();


app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(session({
  secret: "secret-key",
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: process.env.NODE_ENV === "production",
    httpOnly: true
  }
}));

app.use(passport.initialize());
app.use(passport.session());

// ----- MONGODB CONNECTION -----
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log("MongoDB connected"))
.catch(err => console.error("MongoDB connection error:", err));

// ----- PASSPORT CONFIG -----
passport.serializeUser((user, done) => done(null, user.id));
passport.deserializeUser((id, done) => {
  User.findById(id).then(user => done(null, user));
});



// ----- ROUTES -----

// Manual signup
app.post("/signup", async (req, res) => {
  const { name, email, password } = req.body;
  try {
    const existingUser = await User.findOne({ email });
    if(existingUser) return res.status(400).json({ message: "Email already exists" });

    const hashedPassword = await bcrypt.hash(password, 10);
    const user = new User({ name, email, password: hashedPassword });
    await user.save();

    res.status(201).json({ message: "Signup successful" });
  } catch(err) {
    console.error(err);
    res.status(500).json({ message: "Server error" });
  }
});

// Manual login
app.post("/login", async (req, res) => {
  const { email, password } = req.body;
  try {
    const user = await User.findOne({ email });
    if(!user) return res.status(400).json({ message: "User not found" });

    const match = await bcrypt.compare(password, user.password);
    if(!match) return res.status(400).json({ message: "Incorrect password" });

    req.session.userId = user._id;
    res.json({ message: "Login successful" });
  } catch(err) {
    console.error(err);
    res.status(500).json({ message: "Server error" });
  }
});

// Logout
app.get("/logout", (req, res) => {
  req.logout(err => {
    if(err) console.error(err);
    res.redirect("/login.html");
  });
});

// Dashboard (protected route)
app.get("/login", (req, res) => {
  if(!req.user) return res.redirect("/login.html");
  res.send(`Welcome ${req.user.name} to your login`);
});

// ----- START SERVER -----
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
