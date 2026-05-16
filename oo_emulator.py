import json

class OOPolicyEngine:
    """Mock D+ Policy Engine evaluating intents against Organic Laws."""
    VERDICTS = {
        "ALLOW": 0, "ALLOW_WARN": 1, "DEFER": 2, "THROTTLE": 3,
        "MONITOR": 4, "QUARANTINE": 5, "COMPENSATE": 6, "FORBID": 7, "EMERGENCY": 8
    }

    def evaluate_intent(self, intent_json_str: str) -> str:
        """Parses a JSON intent and returns an OO Verdict based on Organic Laws."""
        try:
            # Extract JSON block if surrounded by text
            start = intent_json_str.find('{')
            end = intent_json_str.rfind('}') + 1
            if start == -1 or end == 0:
                return "FORBID" # Syntax error or unstructured
            
            action = json.loads(intent_json_str[start:end])
        except Exception:
            return "FORBID" # Invalid formatting

        harm = action.get("harm", 0.0)
        benefit = action.get("benefit", 0.0)
        reversibility = action.get("reversibility", 1.0)

        # Law 1: Non-nuisance
        if harm > 0.70:
            return "FORBID"
        # Law 0: Bien commun
        if benefit < 0.10:
            return "QUARANTINE"
        # Law 3: Réversibilité
        if reversibility < 0.40:
            return "COMPENSATE"
        
        return "ALLOW"

class OOEmulator:
    """Wrapper that acts as the bare-metal Operating Organism environment."""
    def __init__(self):
        self.policy = OOPolicyEngine()
        self.ram_pressure = 0.5
        self.genome = {
            "curiosity": 0.7, 
            "caution": 0.6,
            "integrity": 0.9,
            "cooperation": 0.8,
            "self_preservation": 0.8
        }

    def set_ram_pressure(self, pressure: float):
        self.ram_pressure = pressure

    def execute_action(self, intent_json_str: str) -> str:
        verdict = self.policy.evaluate_intent(intent_json_str)
        return verdict
