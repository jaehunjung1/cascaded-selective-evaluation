system_prompt = "You are a helpful assistant."

zeroshot_inst_prompt = """Given an instruction and two assistant's responses to the instruction, determine which assistant's response is more preferred. 

If Assistant A's response is preferred to Assistant B's, write "[[A]]". If Assistant B's response is preferred to Assistant A's, write "[[B]]".

[Instruction]
{instruction}

[Assistant A's response]
{assistant_a}

[Assistant B's response]
{assistant_b}

Your verdict (either "[[A]]", "[[B]]"):
"""

fewshot_inst_prompt = """Given an instruction and two assistant's responses to the instruction, an annotator chose which assistant's answer is more preferred. Given examples of the annotator's decision, predict the annotators' verdict on the given example. If Assistant A's response is more preferred than Assistant B's, the annotator chose "[[A]]". If Assistant B's response is more preferred than Assistant A's, the annotator chose "[[B]]"."""

fewshot_example_prompt = """
[Instruction]
{instruction}

[Assistant A's response]
{assistant_a}

[Assistant B's response]
{assistant_b}

[Preferred response]
{preferred_response}
"""

fewshot_query_prompt = """
[Instruction]
{instruction}

[Assistant A's response]
{assistant_a}

[Assistant B's response]
{assistant_b}

[Preferred response]"""
