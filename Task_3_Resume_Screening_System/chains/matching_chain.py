import json
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from prompts.match_prompt import match_prompt

class MatchingOutput(BaseModel):
    matched_skills: List[str] = Field(description="A list of skills that matched the Job Description")
    missing_skills: List[str] = Field(description="A list of skills from the Job Description that the candidate is missing")
    match_percentage: int = Field(description="Calculated match percentage (0-100)")

matching_parser = JsonOutputParser(pydantic_object=MatchingOutput)

def get_matching_chain(llm):
    # Inject format instructions into the prompt dynamically
    prompt_with_instructions = match_prompt.partial(
        format_instructions=matching_parser.get_format_instructions()
    )
    # The prompt actually needs {format_instructions} in the template for `.partial` to inject it effectively.
    # To avoid rewriting the prompt file, we can map it via a wrapper or just append it.
    
    # Let's append format instructions to the template dynamically here for robustness
    prompt_with_instructions.template += "\n\n{format_instructions}"
    
    return prompt_with_instructions | llm | matching_parser
