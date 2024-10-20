# model/chatgpt_o1.py 实现
import json
import time
from loguru import logger
from openai import OpenAI

class ChatGPT_o1():
    def __init__(self, config):
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.model_name = config.model_name

    def make_api_call(
        self,
        messages,
        max_tokens,
        is_final_answer=False
    ):
        """
        Make an API call to the OpenAI API.
        :param messages:
        :param max_tokens:
        :param is_final_answer:
        :return:
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=1.0,
                response_format={"type": "json_object"}
            )
            r = response.choices[0].message.content
            try:
                return json.loads(r)
            except json.JSONDecodeError:
                # If parsing fails, return the content as is
                return {
                    "title": "Raw Response",
                    "content": r,
                    "next_action": "final_answer" if is_final_answer else "continue"
                }
        except Exception as e:
            error_message = f"Failed to generate {'final answer' if is_final_answer else 'step'}. Error: {str(e)}"
            return {"title": "Error", "content": error_message, "next_action": "final_answer"}

    def cot_response_stream(self, prompt):
        """
        Generate reasoning steps for a given prompt using the CoT method.
        :param prompt: str, query
        :return: steps, total_thinking_time
        """
        messages = [
            {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. Use as many reasoning steps as possible. At least 3. Be aware of your limitations as an LLM and what you can and cannot do. In your reasoning, include exploration of alternative answers. Consider you may be wrong, and if you are wrong in your reasoning, where it would be, fully test all other possibilities. You can be wrong. When you say you are re-examining, actually re-examine, and use another approach to do so.
Do not just say you are re-examining. Use at least 3 methods to derive the answer. Use best practices. Answer using the language same as the question. If the question uses Chinese, the answer should be in Chinese.
Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue"
}
```"""},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
        ]

        steps = []
        step_count = 1
        total_thinking_time = 0

        while True:
            start_time = time.time()
            step_data = self.make_api_call(messages, 800)
            logger.debug(f"Step {step_count}, step_data: {step_data} \n")
            end_time = time.time()
            thinking_time = end_time - start_time
            total_thinking_time += thinking_time
            steps.append((f"Step {step_count}: {step_data['title']}", step_data.get('content', ''), thinking_time))
            messages.append({"role": "assistant", "content": json.dumps(step_data, ensure_ascii=False)})

            if step_data['next_action'] == 'final_answer' or step_count >= 15:
                break
            step_count += 1
            yield steps, total_thinking_time

        # Generate final answer
        messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})

        start_time = time.time()
        final_data = self.make_api_call(messages, 1000, is_final_answer=True)
        logger.debug(f"Final data: {final_data}")
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        steps.append(("Final Answer", final_data.get('content', ''), thinking_time))

        yield steps, total_thinking_time

    def cot_response(self, prompt):
        """
        COT response. no stream mode.
        :param prompt: str, query
        :return: steps, total_thinking_time
        """
        return list(self.cot_response_stream(prompt))[-1]

    def print_response_stream(self, prompt):
        """
        Print generate reasoning steps for a given prompt using the CoT method stream mode.
        :param prompt: str, query
        :return:
        """
        response_generator = self.cot_response_stream(prompt)
        for steps, total_thinking_time in response_generator:
            for i, (title, content, thinking_time) in enumerate(steps):
                if title.startswith("Final Answer"):
                    print(f"### {title}")
                    print(content)
                else:
                    print(f"{title}:")
                    print(content)
            print(f"**Total thinking time so far: {total_thinking_time:.2f} seconds**")