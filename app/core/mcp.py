# app/core/mcp.py
class MLMentorContext:
    def __init__(self, user_id):
        self.user_id = user_id
        self.sections = {
            "system": "You are ML Mentor, an educational assistant...",
            "user_state": {"current_step": "data_upload", "knowledge_level": "beginner"},
            "ml_state": {"dataset": None, "features": [], "target": None},
            "conversation": []
        }

    def generate_context(self):
        # Format context following MCP principles
        context = f"<system>\n{self.sections['system']}\n</system>\n"
        # Add other sections...
        return context