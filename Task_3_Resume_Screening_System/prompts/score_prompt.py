from langchain_core.prompts import PromptTemplate

score_template = """You are an expert HR Assessment System. Calculate a final score for the candidate based on the matching results.
Apply the following STRICT scoring logic rules:
- Skills account for up to 50% of the total score.
- Experience accounts for up to 30% of the total score.
- Tools account for up to 20% of the total score.
- High match overall -> Assign a score between 80-100
- Medium match overall -> Assign a score between 50-79
- Low match overall -> Assign a score <50
- CRITICAL: If the candidate has less years of experience than explicitly required in the Job Description, you MUST deduct at least 25 points from their final score.
- Calculate the points logically before outputting the final sum.

Extracted Data:
{extracted_data}

Matching Data:
{matched_data}

You must adhere to the following very strict rules:
- Do NOT assume any information not present in the input
- Return output in JSON format only
- Do not add extra text outside JSON text
- Output key must be exactly "score" (integer).
"""

score_prompt = PromptTemplate(
    input_variables=["extracted_data", "matched_data"],
    template=score_template
)
