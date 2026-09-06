"""Score generations with the official IFStruct validator."""

from __future__ import annotations

from ifstruct.dataset import IfStructExample
from ifstruct.validator import validate_response


def score_example(example: IfStructExample, response: str) -> dict:
    result = validate_response(
        response=response,
        json_schema=example.json_schema,
        top_level_count=example.top_level_count,
        require_no_commentary=example.require_no_commentary,
        output_format=example.output_format,
        top_level_key=example.top_level_key,
        require_wrapper_key=example.require_wrapper_key,
        require_code_block=example.require_code_block,
    )
    return {
        "passed": result.passed,
        "score": result.score,
        "errors": result.errors,
        "details": result.details,
    }
