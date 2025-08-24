import logging
import json
from typing import Dict, Any, Optional
import google.generativeai as genai
from app.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with Gemini API"""
    
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
    async def generate_sql(self, prompt: str) -> Dict[str, Any]:
        """Generate SQL from natural language"""
        try:
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"LLM SQL generation error: {e}")
            return {
                "sql": "",
                "error": str(e)
            }
    
    async def generate_explanation(self, prompt: str) -> Dict[str, Any]:
        """Generate explanation for results"""
        try:
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"LLM explanation error: {e}")
            return {
                "summary": "Unable to generate explanation",
                "error": str(e)
            }
    
    async def detect_ambiguity(self, question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Detect ambiguities in user question"""
        prompt = f"""Analyze this question for ambiguities that need clarification.

Question: {question}
Available columns: {', '.join(schema.keys())}

Identify any ambiguous terms that could have multiple interpretations.

Return as JSON:
{{
    "has_ambiguity": true/false,
    "ambiguities": [
        {{
            "term": "ambiguous term",
            "possible_meanings": ["meaning1", "meaning2"],
            "suggested_clarification": "clarifying question"
        }}
    ]
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Ambiguity detection error: {e}")
            return {"has_ambiguity": False, "ambiguities": []}