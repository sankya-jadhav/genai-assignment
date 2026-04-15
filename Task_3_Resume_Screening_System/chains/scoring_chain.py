from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from prompts.score_prompt import score_prompt

class ScoringOutput(BaseModel):
    score: int = Field(description="The calculated final score (0-100)")

scoring_parser = JsonOutputParser(pydantic_object=ScoringOutput)

def get_scoring_chain(llm):
    prompt_with_instructions = score_prompt.partial()
    prompt_with_instructions.template += "\n\n{format_instructions}"
    prompt_with_instructions = prompt_with_instructions.partial(
        format_instructions=scoring_parser.get_format_instructions()
    )
    return prompt_with_instructions | llm | scoring_parser
