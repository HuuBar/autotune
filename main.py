import os

from langchain_core.messages import HumanMessage

from agent.agent import AutoTuneAgent
from llm.request_llm import RequestLLM
from utils.config import global_config


def llm_test(llm: str):
    llm_kwargs = global_config.get("llm", {}).get(llm)
    req_llm = RequestLLM(role="test", **llm_kwargs)
    resp = req_llm._call([
        HumanMessage(content="你是谁")
    ])
    print(resp)


if __name__ == '__main__':
    # LLM 调用测试
    # llm_test(llm="dsv4")

    AutoTune = AutoTuneAgent()
    AutoTune.run()