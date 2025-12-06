import fetch from "node-fetch";

export async function handler(event, context) {
  try {
    const body = JSON.parse(event.body || "{}");
    const userMessage = body.message;

    if (!userMessage) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: "Missing 'message' field." })
      };
    }

    // Call Groq API
    const response = await fetch(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${process.env.GROQ_API_KEY}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: "llama3-8b-8192",
          messages: [
            { role: "system", content: "You are a friendly statistics tutor." },
            { role: "user", content: userMessage }
          ]
        })
      }
    );

    const data = await response.json();

    // Extract the actual message text
    const botReply = data?.choices?.[0]?.message?.content || "(No response)";

    return {
      statusCode: 200,
      headers: { "Access-Control-Allow-Origin": "*" }, // CORS for Netlify
      body: JSON.stringify({ message: botReply })
    };
  } catch (err) {
    console.error("Error:", err);
    return {
      statusCode: 500,
      headers: { "Access-Control-Allow-Origin": "*" },
      body: JSON.stringify({ error: "Server error." })
    };
  }
}
