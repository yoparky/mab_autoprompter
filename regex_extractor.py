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


def main():
    """
    Main function to test the extract_demarcated_string function.
    """
    # Define standard demarcators for testing
    thought_start_demarcator = "---THOUGHT_PROCESS_START---"
    thought_end_demarcator = "---THOUGHT_PROCESS_END---"
    answer_start_demarcator = "---ANSWER_START---"
    answer_end_demarcator = "---ANSWER_END---"

    # Test Case 1: Typical LLM response with thought process and answer
    llm_response_example_1 = """
    This is some preamble from the LLM.

    ---THOUGHT_PROCESS_START---
    1. First, I considered the input.
    2. Then, I broke down the problem.
    3. Finally, I formulated the solution.
    This process ensures accuracy.
    ---THOUGHT_PROCESS_END---

    After careful consideration, here is the result:

    ---ANSWER_START---
    The primary colors are red, yellow, and blue.
    These colors can be mixed to create other colors.
    ---ANSWER_END---

    I hope this helps!
    """
    print("--- Test Case 1: Standard Extraction ---")
    thought_content_1 = extract_demarcated_string(
        llm_response_example_1,
        thought_start_demarcator,
        thought_end_demarcator
    )
    print(f"Extracted Thought Process:\n'{thought_content_1}'\n")

    answer_content_1 = extract_demarcated_string(
        llm_response_example_1,
        answer_start_demarcator,
        answer_end_demarcator
    )
    print(f"Extracted Answer:\n'{answer_content_1}'\n")

    # Test Case 2: Custom demarcators with regex special characters
    custom_start = "[SECTION_BEGIN:DATA*+?]"
    custom_end = "[SECTION_END:DATA*+?]"
    llm_response_example_2 = f"""
    Some other text.
    {custom_start}
    This is the important data payload.
    It might contain special characters like . ^ $ * + ? {{ }} [ ] \ | ( )
    And it spans multiple lines.
    {custom_end}
    More text.
    """
    print("--- Test Case 2: Custom Demarcators with Special Characters ---")
    custom_data_2 = extract_demarcated_string(
        llm_response_example_2,
        custom_start,
        custom_end
    )
    print(f"Extracted Custom Data:\n'{custom_data_2}'\n")

    # Test Case 3: Demarcators not found in the response
    llm_response_example_3 = "This response has no special tags for the answer."
    print("--- Test Case 3: Demarcators Not Found ---")
    missing_content_3 = extract_demarcated_string(
        llm_response_example_3,
        answer_start_demarcator,
        answer_end_demarcator
    )
    print(f"Content when tags are missing:\n'{missing_content_3}'\n")

    # Test Case 4: Empty content between demarcators
    llm_response_example_4 = f"{answer_start_demarcator}\n   \n{answer_end_demarcator}"
    print("--- Test Case 4: Empty or Whitespace-Only Content ---")
    empty_data_4 = extract_demarcated_string(
        llm_response_example_4,
        answer_start_demarcator,
        answer_end_demarcator
    )
    print(f"Content when section is empty or only whitespace:\n'{empty_data_4}'\n")

    # Test Case 5: Demarcators directly adjacent to content (no newlines)
    llm_response_example_5 = f"{answer_start_demarcator}Adjacent content.{answer_end_demarcator}"
    print("--- Test Case 5: Content Adjacent to Demarcators ---")
    adjacent_content_5 = extract_demarcated_string(
        llm_response_example_5,
        answer_start_demarcator,
        answer_end_demarcator
    )
    print(f"Extracted Adjacent Content:\n'{adjacent_content_5}'\n")

    # Test Case 6: Only the demarcated section exists in the string
    llm_response_example_6 = f"{answer_start_demarcator}Only this content exists.{answer_end_demarcator}"
    print("--- Test Case 6: Only Demarcated Section in String ---")
    only_section_content_6 = extract_demarcated_string(
        llm_response_example_6,
        answer_start_demarcator,
        answer_end_demarcator
    )
    print(f"Extracted Content (Only Section):\n'{only_section_content_6}'\n")
    
    # Test Case 7: Nested-like structures (should grab the outermost valid pair non-greedily)
    # The current non-greedy `(.*?)` will correctly handle the first valid pair.
    llm_response_example_7 = f"""
    {answer_start_demarcator}
    Outer content start.
    {answer_start_demarcator} Inner content. {answer_end_demarcator}
    Outer content end.
    {answer_end_demarcator}
    """
    print("--- Test Case 7: Nested-like Demarcators (Non-Greedy Check) ---")
    nested_content_7 = extract_demarcated_string(
        llm_response_example_7,
        answer_start_demarcator,
        answer_end_demarcator
    )
    # Expected: "Outer content start.\n    ---ANSWER_START--- Inner content. ---ANSWER_END---\n    Outer content end."
    print(f"Extracted Content from Nested-like Structure:\n'{nested_content_7}'\n")


if __name__ == '__main__':
    main()
