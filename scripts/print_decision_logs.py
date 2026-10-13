from pathlib import Path

import duckdb

from sca.llm.client import LLMClient
from sca.sim.environment import EchelonState
from sca.systems.llm_base import OrderDecision
from sca.systems.multi_agent_tools import MultiAgentToolsSystem
from sca.systems.single_agent import SingleAgentSystem

db_path = Path("results/demo_logs.duckdb")
if db_path.exists():
    db_path.unlink()

client = LLMClient(mock_mode=True, db_path=db_path)
client.mock.set_response("CURRENT LOCAL STATE", OrderDecision(quantity=98.0, reasoning="Balanced replenishment aligned with deterministic forecast"))

state = EchelonState(
    echelon="retailer",
    step=3,
    inventory=120.0,
    backlog=0.0,
    pipeline=40.0,
    incoming_shipments=25.0,
    current_demand=100.0,
    last_order=95.0,
    lead_time=2,
    history_window=[],
)

# 1. Deterministic Mode Run
sys_det = SingleAgentSystem(name="single_agent", llm_client=client, db_path=db_path, run_id="run_det_demo", tool_mode="deterministic")
orders_det = sys_det.decide_orders({"retailer": state})

conn = duckdb.connect(str(db_path))
log_det = conn.execute("""
    SELECT system_name, echelon, step_number, prompt_version, tool_recommendation, order_quantity, reasoning, retry_count, fallback
    FROM agent_calls WHERE run_id = 'run_det_demo'
""").fetchone()

# 2. LLM Function Calling Mode Run
sys_fc = MultiAgentToolsSystem(name="multi_agent_tools", config={"aggregation_rule": "median", "tool_mode": "llm_function_calling"}, db_path=db_path, run_id="run_fc_demo", tool_mode="llm_function_calling")
orders_fc = sys_fc.decide_orders({"retailer": state})

logs_fc = conn.execute("""
    SELECT tool_name, expected_tool, is_correct_tool, is_parameter_valid
    FROM tool_calls WHERE run_id = 'run_fc_demo' ORDER BY created_at ASC
""").fetchall()
conn.close()

print("=" * 80)
print("DECISION LOG 1: DETERMINISTIC MODE ('deterministic tools plus LLM decision and negotiation')")
print("=" * 80)
print(f"System:             {log_det[0]}")
print(f"Echelon:            {log_det[1]}")
print(f"Step:               {log_det[2]}")
print(f"Prompt Version:     {log_det[3]}")
print(f"Tool Recommendation: {log_det[4]:.1f}")
print(f"Decided Order:      {log_det[5]:.1f}")
print(f"LLM Reasoning:      \"{log_det[6]}\"")
print(f"Retries / Fallback: {log_det[7]} / {log_det[8]}")
print("=" * 80)

print("\n" + "=" * 80)
print("DECISION LOG 2: LLM FUNCTION CALLING MODE ('llm_function_calling')")
print("=" * 80)
print(f"System:             {sys_fc.name}")
print("Echelon:            retailer")
print("Step:               3")
print(f"Decided Order:      {orders_fc['retailer']:.1f}")
print("Tool Execution Trace (Evaluated against expected-tool rule):")
for idx, (t, exp, corr, val) in enumerate(logs_fc, 1):
    print(f"  [{idx}] Invoked: '{t:<16}' | Expected: '{exp:<16}' | Correct: {corr} | Valid Parameter: {val}")
print("=" * 80)
