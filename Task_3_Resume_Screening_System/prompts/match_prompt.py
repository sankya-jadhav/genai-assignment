from langchain_core.prompts import PromptTemplate

match_template = """You are an expert HR Analyst. Compare the extracted candidate data with the job description.
Identify the matched skills and missing skills. Calculate a general match percentage based on the alignment of skills.

Extracted Data from Resume:
{extracted_data}

Job Description:
{job_description}

You must adhere to the following very strict rules:
- Only compare given data
- Do NOT assume any information not present in the input
- Return output in JSON format only
- Do not add extra text outside JSON text
- Output keys must be exactly "matched_skills" (list of strings), "missing_skills" (list of strings), and "match_percentage" (integer from 0 to 100).
"""

match_prompt = PromptTemplate(
    input_variables=["extracted_data", "job_description"],
    template=match_template
)
