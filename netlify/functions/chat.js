// File: netlify/functions/chat.js
import fetch from "node-fetch";

export async function handler(event, context) {
  try {
    // Parse incoming request
    const body = JSON.parse(event.body || "{}");
    const userMessage = body.message?.trim();

    if (!userMessage) {
      return {
        statusCode: 400,
        body: JSON.stringify({ message: "(No message provided)" })
      };
    }

    // Call Groq API with groq/compound-mini
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.GROQ_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "groq/compound-mini",
        messages: [
          { role: "system", content: "You are a friendly statistics tutor." },
          { role: "user", content: userMessage }
        ]
      })
    });

    const data = await response.json();

    // Extract bot reply
    const botReply = data?.choices?.[0]?.message?.content || "(No response)";

    return {
      statusCode: 200,
      body: JSON.stringify({ message: botReply })
    };

  } catch (err) {
    console.error("Error contacting Groq API:", err);
    return {
      statusCode: 500,
      body: JSON.stringify({ message: "(Groq API error)" })
    };
  }
}
