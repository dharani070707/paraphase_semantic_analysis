const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

// Health check
app.get("/", (req, res) => {
  res.send("Backend running 🚀");
});

// MAIN API
app.post("/analyze", (req, res) => {
  const { text1, text2 } = req.body;

  if (!text1 || !text2) {
    return res.status(400).json({ error: "Missing input" });
  }

  const response = {
    similarity: 0.98,
    paraphrase: true,
  };

  res.json(response);
});

app.listen(5000, () => {
  console.log("Server running on http://localhost:5000");
});
