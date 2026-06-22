import re
import json
import random
import logging

from typing import Tuple

from config import *
from utils import get_llm_result
from pipeline.dqs.diverse_query import DiverseQuery
from prompt.diverse_instruct_prompt import (
    INSTRUCTION_TOOL_DESC_TEMPLATE,
    INSTRUCTION_TOOL_CALL_TEMPLATE,
    TOOL_INSTRUCT_FN_NAME,
    TOOL_INSTRUCT_FN_ARGS,
    TOOL_INSTRUCT_FN_RESULT,
    TOOL_INSTRUCT_FN_EXIT,
    TOOL_EVOLVE_REQUIREMENT,
    TOOL_EVOLVE_EXAMPLE,
    TOOL_EVOLVE_PATTERN,
    TOOL_EVOLVE_PROMPT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolData:
    def __init__(
        self,
        llm,
        tool_desc_path: str,
        embedding_path: str = EMBED_MODEL_PATH,
        tevo_prompt_template: str = TOOL_EVOLVE_PROMPT,
        tevo_num: int = 10,
    ):
        self.llm = llm,
        self.tevo_num = tevo_num,
        self.tevo_prompt_template = tevo_prompt_template
        self.embedding_path = embedding_path
        # keywords in tool_map
        self.tool_desc_key = "desc"
        self.tool_param_key = "param"
        self.tool_require_key = "req"
        # build tool_map
        self.tool_map = self._init_tools_map(tool_desc_path)

    def _init_tools_map(self, tool_desc_path: str) -> dict:
        '''
        extract tool info from json file into tools(list of dicts) and tool_map(dict of name-desc pairs)
        return:
            tool_map: dict{tool_name: {"desc":str, "param":dict, "req":list}}
        '''
        with open(tool_desc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tool_map = {}
        for tool in data:
            func = tool.get("description", {}).get("function", {})
            tool_map[func["name"]] = {
                self.tool_desc_key: func.get("description", ""),
                self.tool_param_key: func.get("parameters", {}).get("properties", {}),
                self.tool_require_key: func.get("parameters", {}).get("required", [])
            }
        return tool_map

    def fill_tools(self, tools: list[str], min_tool_num: int, is_random: bool = False):
        '''
        modify arg (tools) to fill it up by remained tools in self.tool_map until min_tool_num
        '''
        if len(tools) >= min_tool_num:
            return
        fill_size = min_tool_num - len(tools)
        extra = [name for name in self.tool_map.keys() if name not in tools]
        # fill tools randomly
        if is_random:
            tools += random.sample(extra, fill_size)
            return
        # fill tools by description similarity
        row = [self.tool_map[name]["desc"] for name in tools]
        col = [self.tool_map[name]["desc"] for name in extra]
        remain_idx = DiverseQuery.similarity_filter(self.embedding_path, row, col, len(col) - fill_size, 0)
        if len(remain_idx) > fill_size:
            tools += [extra[idx] for idx in random.sample(remain_idx, fill_size)]
        else:
            tools += [extra[idx] for idx in remain_idx]

    def get_tool_description(self, tool_name: str, evolve_name = "", evolve_desc = "") -> str:
        '''
        args:
            tool_name: str
            evolve_name(opt): name of evolved tool
            evolve_desc(opt): desc of evolved tool
        return:
            tool_description: str
        '''
        if tool_name not in self.tool_map:
            return ""
        tool_desc = self.tool_map.get(tool_name, {}).get(self.tool_desc_key, "")
        tool_params = self.tool_map.get(tool_name, {}).get(self.tool_param_key, {})
        required_params = self.tool_map.get(tool_name, {}).get(self.tool_require_key, [])
        return INSTRUCTION_TOOL_DESC_TEMPLATE.format(
            tool_name = evolve_name if evolve_name else tool_name, 
            tool_description = evolve_desc if evolve_desc else tool_desc,
            tool_params = str(tool_params),
            required_params = str(required_params)
        )

    def get_tool_instruction(self, tool_list: list, tool_evo = {}) -> Tuple[str, list]:
        """
        args:
            tool_list: list[tool_name(str)]
            tool_evo(opt): dict{tool_name: {"name": [evolve_names], "description": [evolve_descs]}}
                      if dict is empty instruction_tool_descs will be formated by original tool_list
        return:
            instruction_tool_call: formated tool_instruction string
            tool_name_list: list[tool_name]
        """
        # build evolve_tool_list and format tool_descs for instruction
        tool_name_list = []
        instruction_tool_descs = ""
        for tool in tool_list:
            evolve_name, evolve_desc = "", ""
            if tool_evo:
                # randomly pick evolved tool names and descriptions from tool_evo
                evolve_names = tool_evo.get(tool, {}).get("name", [])
                evolve_descs = tool_evo.get(tool, {}).get("description", [])
                if evolve_names and evolve_descs and (len(evolve_names) == len(evolve_descs)):
                    rand_idx = random.randint(0, len(evolve_names) - 1)
                    evolve_name = evolve_names[rand_idx]
                    evolve_desc = evolve_descs[rand_idx]
                    tool_name_list.append(evolve_name)
                else:
                    tool_name_list.append(tool)
            else:
                tool_name_list.append(tool)
            instruction_tool_descs += self.get_tool_description(tool, evolve_name, evolve_desc)
        # format tool_instruction
        tool_instruction = INSTRUCTION_TOOL_CALL_TEMPLATE.format(
            tool_descs = instruction_tool_descs,
            tool_names = str(tool_name_list),
            fn_name = TOOL_INSTRUCT_FN_NAME,
            fn_args = TOOL_INSTRUCT_FN_ARGS,
            fn_result = TOOL_INSTRUCT_FN_RESULT,
            fn_exit = TOOL_INSTRUCT_FN_EXIT
        )
        return tool_instruction, tool_name_list

    def _extract_tool_evo_res(self, input_list: list) -> list:
        res = []
        for it in input_list:
            match = re.search(TOOL_EVOLVE_PATTERN, it)
            if match:
                name = match.group(1).strip()
                desc = match.group(2).strip()
                res.append(f"{name}:{desc}")
        return res

    def _format_tool_evo_prompt(self, template, num: int, tools: str) -> list:
        prompt_value = template.invoke(
            {
                "num": str(num),
                "tools": tools,
                "requirement": TOOL_EVOLVE_REQUIREMENT,
                "example": TOOL_EVOLVE_EXAMPLE,
            }
        )
        return prompt_value.messages

    def tool_evolve(self, output_path: str, size = 4) -> dict:
        """
        args:
            output_path: path to save tool_evo
            size: number of remain size for filter
        return:
            tool_evo: dict{"raw_tool": {"name": list, "description": list}}
        """
        logger.debug('=== Tool Evolve===')
        tool_evo = {}
        for k, v in self.tool_map.items():
            # generate tools from llm
            desc = v["desc"]
            tools = f"名称:{k}; 描述:{desc}"
            messages = self._format_tool_evo_prompt(self.tevo_prompt_template, self.tevo_num, tools)
            res = get_llm_result(self.llm, messages, r'\d+(?![\n\s])\W+')
            extract_res = self._extract_tool_evo_res(res)
            # filter out dissimilar tools
            remain_idx = DiverseQuery.similarity_filter(
                self.embedding_path, [f"{k}:{desc}"], extract_res, len(extract_res) - size, 0
            )
            # build tool_evo dict
            name_list, desc_list = [k], [desc]
            for idx in remain_idx:
                name_list.append(extract_res[idx].split(":")[0].strip())
                desc_list.append(extract_res[idx].split(":")[1].strip())
            tool_dict = {}
            tool_dict["name"] = name_list
            tool_dict["description"] = desc_list
            tool_evo[k] = tool_dict
        # save tool_evo to json file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tool_evo, f, indent=4, ensure_ascii=False)
        return tool_evo
