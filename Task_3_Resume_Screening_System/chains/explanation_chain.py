from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from prompts.explain_prompt import explain_prompt

class ExplanationOutput(BaseModel):
    explanation: str = Field(description="A factual concise explanation of the candidate's score")

explanation_parser = JsonOutputParser(pydantic_object=ExplanationOutput)

def get_explanation_chain(llm):
    prompt_with_instructions = explain_prompt.partial()
    prompt_with_instructions.template += "\n\n{format_instructions}"
    prompt_with_instructions = prompt_with_instructions.partial(
        format_instructions=explanation_parser.get_format_instructions()
    )
    return prompt_with_instructions | llm | explanation_parser
