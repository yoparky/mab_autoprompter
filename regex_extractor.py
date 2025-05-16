import re

def extract_demarcated_string(llm_raw_response: str, start_demarcator: str, end_demarcator: str) -> str | None:

    escaped_start_demarcator = re.escape(start_demarcator)
    escaped_end_demarcator = re.escape(end_demarcator)

    answer_pattern = rf"{escaped_start_demarcator}(.*?){escaped_end_demarcator}"
    match = re.search(answer_pattern, llm_raw_response, re.DOTALL)

    if match:
        extracted_string = match.group(1).strip()
        return extracted_string if extracted_string else ""
    return ""
