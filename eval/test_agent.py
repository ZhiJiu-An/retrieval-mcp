"""Agent 层门禁：工具面好不好用。

这一组指标最容易被忽略：检索 Recall 涨 5 个点，如果 description 写得差，
模型仍要多调 3 次工具才收敛，端到端反而更慢更贵（DESIGN §13.1）。

口径：`simulate_agent` 是 Agent 行为的**代理**，不是真 LLM（见 runner.py 的说明）。
"""

from __future__ import annotations

GATE_SUCCESS_RATE = 0.85
GATE_MEDIAN_CALLS = 2.5
GATE_MEAN_CHARS = 12_000

#: 无答案 / 应被拒 / 同义改写不参与「任务成功率」——它们衡量的是别的东西。
_EXCLUDE = {"paraphrase", "no_answer", "denied", "superseded"}


def test_agent_finds_the_answer(agent_report):
    core = agent_report.subset(exclude=_EXCLUDE)
    assert core.accuracy_at_1 >= GATE_SUCCESS_RATE, (
        f"任务成功率={core.accuracy_at_1}，失败用例：{[c.case.id for c in core.failures()]}"
    )


def test_agent_converges_in_few_calls(agent_report):
    core = agent_report.subset(exclude=_EXCLUDE)
    assert core.median_calls <= GATE_MEDIAN_CALLS, f"工具调用中位数={core.median_calls}"


def test_context_cost_stays_under_budget(agent_report):
    core = agent_report.subset(exclude=_EXCLUDE)
    assert core.mean_chars <= GATE_MEAN_CHARS, f"平均上下文字符数={core.mean_chars}"


def test_negative_cases_do_not_drag_the_agent_into_loops(agent_report):
    """无答案的问题不应该让 Agent 反复重试 —— hint 必须足以让它停下来。"""
    negative = agent_report.subset(tags={"no_answer"})
    assert negative.median_calls <= 2, f"无答案用例的工具调用中位数={negative.median_calls}"
