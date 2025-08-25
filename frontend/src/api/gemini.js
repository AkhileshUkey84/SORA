import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.REACT_APP_GEMINI_API_KEY);

export async function askGemini(question, context) {
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  // Limit to first 50 rows to avoid token issues
  const limitedContext = context.slice(0, 50);

  const prompt = `
Question: ${question}
Context: ${JSON.stringify(limitedContext)}
`;

  const result = await model.generateContent(prompt);

  try {
    return JSON.parse(result.response.text());
  } catch {
    // Fallback if API returns plain text
    return { generatedSQL: null, results: [] };
  }
}
