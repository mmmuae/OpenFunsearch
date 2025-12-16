# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

# import llm
import numpy as np
from funsearch import evaluator
from funsearch import programs_database
import re
import autopep8

def reformat_to_two_spaces(code: str) -> str:
    # Regular expression to match leading spaces at the beginning of each line
    pattern = r"^\s+"

    # Function to replace leading spaces with two spaces per indentation level
    def replace_with_two_spaces(match):
        space_count = len(match.group(0))
        return ' ' * (2 * (space_count // 4))  # Assumes original indentation was 4 spaces

    # Split the code into lines, reformat each line, and join back into a single string
    reformatted_lines = [
        re.sub(pattern, replace_with_two_spaces, line)
        for line in code.splitlines()
    ]
    return "\n".join(reformatted_lines)

def post_process(code: str) -> str:
    # Define a list of patterns to remove
    patterns = [
        r'\[/INST\]',
        r'>\[INST\]',
        r'<s>',
        r'<\s>',
        r'\[PYTHON\]\n```',
        r'```\n\[/PYTHON\]',
        r'\[PYTHON\]',
        r'\[/PYTHON\]',
    ]

    for pattern in patterns:
        code = re.sub(pattern, '', code)
    return code

class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int, model, log_path=None, model_type='gpt', function_to_evolve='priority') -> None:
    self._samples_per_prompt = samples_per_prompt
    self.model = model
    self.prompt_count = 0
    self.log_path = log_path
    self.model_type = model_type
    self.function_to_evolve = function_to_evolve

  def _draw_sample(self, prompt: str) -> str:
  
    # Get a raw completion string from the configured backend.
    if self.model_type in ('llm', 'gpt'):
      resp = self.model.prompt(prompt)
      text_attr = getattr(resp, "text", None)
      if callable(text_attr):
        output_text = text_attr()
      elif isinstance(text_attr, str):
        output_text = text_attr
      else:
        output_text = str(resp)
    else:
      output = self.model(
          "<s>[INST] " + prompt + " [/INST]",
          max_tokens=4096,
          stop=["</s>"],
          echo=True
      )
      output_text = output['choices'][0]['text']
  
    # Saves full response and prompt for debugging purposes
    with open('last_full_responses.txt', 'a') as file_eval:
      file_eval.write(f"PRE POSTPROCESSING RESPONSE {self.prompt_count}\n{output_text}\n")
      file_eval.flush()
  
    with open('last_prompts.txt', 'a') as file_eval:
      file_eval.write(f"Prompt {self.prompt_count}\n{prompt}\n")
      file_eval.flush()
  
    # Code extraction heuristics (tries multiple common code block formats)
    marker_fun = '```@funsearch.run\n'
    idx_fun = output_text.find(marker_fun)
    code_start = (idx_fun + 3) if idx_fun != -1 else -1
  
    code_start_md = output_text.find('```\n')
    code_end_md = output_text.find('```', code_start_md + 3) if code_start_md != -1 else -1
    code_start_py = output_text.find('```python\n')
    code_end_py = output_text.find('```', code_start_py + 3) if code_start_py != -1 else -1
    # Use dynamic function name instead of hardcoded 'priority_v2'
    code_start_def = output_text.find(f'def {self.function_to_evolve}_v')
  
    if code_start_py != -1 and code_end_py != -1:
      response = output_text[code_start_py + len('```python\n'):code_end_py]
    elif code_start_md != -1 and code_end_md != -1:
      response = output_text[code_start_md + len('```\n'):code_end_md]
    elif code_start != -1:
      response = output_text[code_start:]
    elif code_start_def != -1:
      response = output_text[code_start_def:]
    else:
      response = output_text
  
    # Saves after-process response for debugging purposes
    with open('last_processed_responses.txt', 'a') as file_eval:
      file_eval.write(f"AFTER POSTPROCESSING RESPONSE {self.prompt_count}\n{response}\n")
      file_eval.flush()
  
    response = post_process(response)
    response = autopep8.fix_code(response, options={'indent_size': 2})
  
    with open('last_eval.txt', 'a') as file_eval:
      file_eval.write(f"FINAL RESPONSE\n{response}\n")
      file_eval.flush()
  
    if response:
      self._log(prompt, response, self.prompt_count)
    self.prompt_count += 1
    return response

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

  def _log(self, prompt: str, response: str, index: int):
    if self.log_path is not None:
      with open(self.log_path / f"prompt_{index}.log", "a") as f:
        f.write(prompt)
      with open(self.log_path / f"response_{index}.log", "a") as f:
        f.write(str(response))


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      model: LLM,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = model

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    prompt = self._database.get_prompt()
    samples = self._llm.draw_samples(prompt.code)
    # This loop can be executed in parallel on remote evaluator machines.

    with open('last_eval.txt', 'a') as file_eval:   #PVD: show when output is valid according to parser
      file_eval.write(f"SAMPLES\n{samples}\n")
      file_eval.flush()

    for sample in samples:
      chosen_evaluator = np.random.choice(self._evaluators)
      chosen_evaluator.analyse(
          sample, prompt.island_id, prompt.version_generated)
