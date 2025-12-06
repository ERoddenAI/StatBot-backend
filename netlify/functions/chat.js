import fetch from "node-fetch";

export async function handler(event, context) {
  try {
    // Parse incoming request
    const body = JSON.parse(event.body || "{}");
    const userMessage = body.message;

    if (!userMessage) {
      return { 
        statusCode: 400, 
        body: JSON.stringify({ message: "(No message provided)" }) 
      };
    }

    // Check if API key is present
    if (!process.env.GROQ_API_KEY) {
      console.error("GROQ_API_KEY is missing!");
      return { 
        statusCode: 500, 
        body: JSON.stringify({ message: "(Server misconfigured: missing API key)" }) 
      };
    }

    console.log("Sending request to Groq API for user message:", userMessage);

    // Call Groq API
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.GROQ_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "groq/compound-mini", // make sure your account can access this
        messages: [
          { role: "system", content: "You are a friendly statistics tutor." },
          { role: "user", content: userMessage }
        ]
      })
    });

    const data = await response.json();
    console.log("Raw GROQ response:", data);

    const botReply = data?.choices?.[0]?.message?.content || "(No response)";

    return {
      statusCode: 200,
      body: JSON.stringify({ message: botReply })
    };

  } catch (err) {
    console.error("Error in Netlify function:", err);
    return { 
      statusCode: 500, 
      body: JSON.stringify({ message: "(Error contacting API)" }) 
    };
  }
}
