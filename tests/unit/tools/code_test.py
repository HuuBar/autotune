import os

from tools.code.ast_helper import *
from tools.code.editor import EditorTools
from tools.code.navigation import NavigationTools
from tools.code.executor import SandboxTools
from tools.execute.execute_bash import ExecuteBashTool
from tools.execute.execute_git import ExecuteGitTool
from tools.search.web_search import WebSearchTools
from utils.config import global_config


def test():
    # grep_search = NavigationTools().create_grep_search_tool()
    # grep_res = grep_search.invoke({
    #     "query": "similarity_filter",
    # })
    # print(grep_res)

    # find_definition = NavigationTools().create_find_definition_tool()
    # find_res = find_definition.invoke({
    #     "target_name": "get_tool_",
    #     "search_dir": "test/",
    #     "exact_match": False,
    # })
    # print(find_res)

    # read_code_block = NavigationTools().create_read_code_block_tool()
    # read_res = read_code_block.invoke({
    #     "file_path": "test/tool_data.py",
    #     "start_line": 108,
    #     "end_line": 146,
    # })
    # print(read_res)

    # get_file_skeleton = NavigationTools().create_get_file_skeleton_tool()
    # skeleton_res = get_file_skeleton.invoke({
    #     "file_path": "test/tool_data.py",
    # })
    # print(skeleton_res)

    inspect_dataframe = NavigationTools().create_inspect_dataframe_tool()
    df_res = inspect_dataframe.invoke({
        "file_path": "cluster/quantify_result/cluster_coverage_05061118.csv",
    })
    print(df_res)

    # edit_res = edit_code_block.invoke({
    #     "file_path": "/home/lxy/metro_arrive/cluster/clusters.py",
    #     "start_line": 111,
    #     "end_line": 116,
    #     "new_code": "# 7. 兜底策略：处理边缘噪声点\n        if n_clusters > 0:\n            noise_mask = (cluster_labels == -1)\n            noise_count = noise_mask.sum()\n            total_count = len(cluster_labels)\n            if noise_count > 0 and (noise_count / total_count) < 0.15:\n                # 噪声点比例 < 15%，将其分配给最近的聚类中心\n                noise_indices = np.where(noise_mask)[0]\n                for idx in noise_indices:\n                    point_feature = feature_matrix[idx]\n                    # 计算到每个聚类中心的距离\n                    min_dist = float('inf')\n                    assigned_label = 0\n                    for i in range(n_clusters):\n                        cluster_mask = (cluster_labels == i)\n                        center_feature = feature_matrix[cluster_mask].mean(axis=0)\n                        dist = np.linalg.norm(point_feature - center_feature)\n                        if dist < min_dist:\n                            min_dist = dist\n                            assigned_label = i\n                    cluster_labels[idx] = assigned_label\n                logger.info(f\"兜底策略：已将 {noise_count} 个噪声点分配给最近聚类中心\")\n        return {\n            'feature_matrix': feature_matrix,\n            'cluster_labels': cluster_labels,\n            'n_clusters': n_clusters,\n            'cluster_details': cluster_details,\n        }"
    # })
    # print(edit_res)

    # search_res = debug_search.invoke({
    #     "error_message": "How to use __init_subclass__ in python class?",
    #     "source": "stackoverflow"
    # })
    # print(search_res)

    # exe_res = execute_sandbox.invoke({
    #     "command": "python cluster/run.py"
    # })
    # print(exe_res)
