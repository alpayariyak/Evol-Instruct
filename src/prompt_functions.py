depth_instruction = "I want you act as a Prompt Rewriter.\r\n \
					Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
					But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
					Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
					You SHOULD complicate the given prompt using the following method: \r\n\
					{} \r\n\
					You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
					'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"

breadth_instruction = "I want you act as a Prompt Creator.\r\n\
					Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
					This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
					The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
					The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
					'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"

comparison_instruction = "Here are two Instructions to ChatGPT AI, do you think they are equal to each other, which meet the following requirements:\r\n\
					1. They have same constraints and requirements.\r\n\
					2. They have same depth and breadth of the inquiry.\r\n\
					The First Prompt: <Here is first instruction.>\r\n\
					The Second Prompt: <Here is second instruction.>\r\n\
					Your Judgement (Just answer: Equal or Not Equal. No need to explain the reason.):\r\n"





def createBreadthPrompt(instruction):
    prompt = breadth_instruction
    prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Created Prompt#:\r\n"
    return prompt


def createConstraintsPrompt(instruction):
    prompt = depth_instruction.format("Please add one more constraints/requirements into #The Given Prompt#'")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createDeepenPrompt(instruction):
    prompt = depth_instruction.format(
        "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createConcretizingPrompt(instruction):
    prompt = depth_instruction.format("Please replace general concepts with more specific concepts.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createReasoningPrompt(instruction):
    prompt = depth_instruction.format(
        "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createComparisonEliminatorPrompt(before, after):
    prompt = comparison_instruction
    prompt = prompt.replace("<Here is first instruction.>", before)
    prompt = prompt.replace("<Here is second instruction.>", after)
    return prompt
