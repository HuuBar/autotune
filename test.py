import os
import json

from langchain_core.messages import HumanMessage

from llm.request_llm import RequestLLM
from tests.unit.tools.code_test import test
from utils.config import global_config


if __name__ == '__main__':
    llm_config = global_config.get("llm", {})
    llm_kwargs = llm_config.get("dsv4")

    # req_llm = RequestLLM(role="test", **llm_kwargs)
    # resp = req_llm._call([
    #     HumanMessage(content="你是谁")
    # ])
    # print(resp)

    # test()

    # from langgraph.checkpoint.postgres import PostgresSaver
    # from langgraph.store.postgres import PostgresStore
    # from langgraph.store.memory import InMemoryStore
    # conn_string = "postgresql://root:qwe123@localhost:5432/test"
    # with PostgresStore.from_conn_string(conn_string) as store:
    #         store.setup()
    #         store.put(("docs",), "/skills", {"content": "qweasdzxc"})
    #         item = store.get(("docs",), "/skills")
    #         print(item)

    from agent.agent import AutoTuneAgent
    AutoTune = AutoTuneAgent()
    AutoTune.run()
