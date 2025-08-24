import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.REACT_APP_GEMINI_API_KEY);

export async function askGemini(question, context) {
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  const prompt = `
  Question: ${question}
  Context: ${JSON.stringify(context)}
  `;

  const result = await model.generateContent(prompt);
  return JSON.parse(result.response.text());
}
