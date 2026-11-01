import json
import os
import unittest
from unittest.mock import patch

from dataevolver.harness.planner import AgenticRSIPlanner, OpenAICompatiblePlannerBackend


class PlannerWireContractTest(unittest.TestCase):
    def test_http_request_contains_complete_self_contained_decision_contract(self):
        messages = [{"role": "system", "content": "Plan a bounded experiment."},
                    {"role": "user", "content": "No quality evidence is available."}]
        environment = {"DATAEVOLVER_AGENT_BASE_URL": "http://example.invalid/v1",
                       "DATAEVOLVER_AGENT_MODEL": "test-model", "DATAEVOLVER_AGENT_API_KEY": "test-key"}
        with patch.dict(os.environ, environment), patch("urllib.request.urlopen") as send:
            send.return_value.__enter__.return_value.read.return_value = (
                b'{"choices":[{"message":{"content":"{}"}}]}')
            backend = OpenAICompatiblePlannerBackend({})
            backend.complete_json(messages, schema=AgenticRSIPlanner.decision_schema())
            body = json.loads(send.call_args.args[0].data)
        contracts = [json.loads(message["content"].split("\n\n")[-1])["response_schema"]
                     for message in body["messages"]
                     if message["role"] == "system" and '{"response_schema"' in message["content"]]
        self.assertEqual(len(contracts), 1, "Planner schema must reach the model, not just the backend argument")
        schema = contracts[0]
        self.assertTrue(set(schema["required"]) <= set(schema["properties"]))
        self.assertEqual(schema["properties"]["asset_id"], {"type": "string", "minLength": 1})
        self.assertEqual(schema["properties"]["confidence"]["enum"], ["low", "medium", "high"])
        self.assertNotIn('"$ref"', json.dumps(schema))
        self.assertIn("prompt_text", schema["properties"]["strategy_patch"]["oneOf"][0]["properties"])
        self.assertEqual(sum(message["role"] == "system" for message in body["messages"]), 1)
        self.assertTrue(body["messages"][0]["content"].startswith(messages[0]["content"] + "\n\n"))
        self.assertEqual(body["messages"][1:], messages[1:])
        self.assertEqual(messages[0]["content"], "Plan a bounded experiment.")
        self.assertEqual(len(messages), 2)


if __name__ == "__main__":
    unittest.main()
