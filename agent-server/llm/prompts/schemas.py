# llm/prompts/schemas.py

PLAN_EXECUTION_SCHEMA = {
    "type": "execution",
    "steps": [
        {
            "step": 1,
            "model": "registered_model_name",
            "department": "department_name",
            "project": "project_name",
        }
    ],
    "missing_inputs": [],
    "message": "사용자에게 보여줄 자연스러운 한국어 설명",
}


PLAN_KNOWLEDGE_SCHEMA = {
    "type": "knowledge",
    "message": "사용자에게 보여줄 자연스러운 한국어 답변",
}


PLAN_GENERAL_SCHEMA = {
    "type": "general",
    "message": "범용 분석을 수행합니다.",
}


PLAN_OUTPUT_SCHEMA_TEXT = """
For model execution:
{
  "type": "execution",
  "steps": [
    {
      "step": 1,
      "model": "registered_model_name",
      "department": "department_name",
      "project": "project_name"
    }
  ],
  "missing_inputs": [],
  "message": "사용자에게 보여줄 자연스러운 한국어 설명"
}

For clinical knowledge:
{
  "type": "knowledge",
  "message": "사용자에게 보여줄 자연스러운 한국어 답변"
}

For general analysis:
{
  "type": "general",
  "message": "범용 분석을 수행합니다."
}
""".strip()


ALLOWED_PLAN_TYPES = {"execution", "knowledge", "general"}