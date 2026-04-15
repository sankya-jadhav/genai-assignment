from langchain_core.prompts import PromptTemplate

explain_template = """You are an expert HR Assessment System. Generate a concise explanation for the applicant's score based on the matching results and the final score assignment.

Matched Data:
{matched_data}

Assigned Score:
{score_data}

You must adhere to the following very strict rules:
- Be factual and do NOT hallucinate any skills or experience not present.
- Keep the explanation concise (2-4 sentences max).
- Return output in JSON format only.
- Do not add extra text outside JSON text.
- Output key must be exactly "explanation" (string).
- Do not just repeat the score in the explanation string, focus on WHY the score was given.
- Do not prefix the explanation string with "Explanation:" or similar words.
"""

explain_prompt = PromptTemplate(
    input_variables=["matched_data", "score_data"],
    template=explain_template
)
