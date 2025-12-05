import express from "express";
import fetch from "node-fetch";
import cors from "cors";

const app = express();

app.use(cors());
app.use(express.json());

app.post("/chat", async (req, res) => {
  const userMessage = req.body.message;

  if (!userMessage) {
    return res.status(400).send({
      error: "Request body must include a 'message' field."
    });
  }

  try {
    const apiKey = process.env.OPENAI_API_KEY;

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          { role: "user", content: userMessage }
        ]
      })
    });

    const data = await response.json();
    res.send(data);

  } catch (error) {
    console.error("Error calling OpenAI:", error);
    res.status(500).send({ error: "Server error calling OpenAI" });
  }
});

// IMPORTANT: Use Render-assigned port
app.listen(process.env.PORT || 3000, () => {
  console.log("Server running");
});
