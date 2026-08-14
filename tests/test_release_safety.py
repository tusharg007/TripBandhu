from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseSafetyTest(unittest.TestCase):
    def test_dockerignore_excludes_env_from_context(self):
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

        self.assertIn(".env", dockerignore)
        self.assertIn(".env.*", dockerignore)
        self.assertIn("!.env.example", dockerignore)
        self.assertIn("__pycache__/", dockerignore)
        self.assertIn("*.pyc", dockerignore)

    def test_required_mcp_files_exist(self):
        self.assertTrue((ROOT / "mcp_client.py").is_file())
        self.assertTrue((ROOT / "custom_weather_mcp_server.py").is_file())

    def test_dockerfile_installs_and_checks_uvx(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("uv==0.8.13", dockerfile)
        self.assertIn("uvx --version", dockerfile)

    def test_frontend_contains_hitl_and_sanitized_markdown_paths(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "script.js").read_text(encoding="utf-8")

        self.assertIn("New Trip", template)
        self.assertIn("Approve Itinerary", template)
        self.assertIn("Request Revision", template)
        self.assertIn("dompurify@3.2.6", template)
        self.assertIn("/api/travel/resume", script)
        self.assertIn("DOMPurify.sanitize", script)

    def test_quick_load_presets_contract(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "script.js").read_text(encoding="utf-8")

        # HTML elements and attributes
        self.assertIn('class="quick-load"', template)
        self.assertIn('data-prompt="Plan a 7-day cultural and food-focused trip', template)
        self.assertIn('data-prompt="Plan a 5-day luxury and shopping trip', template)
        self.assertIn('data-prompt="Plan a 7-day tropical beach and culture trip', template)
        self.assertIn('data-prompt="Create a relaxed 4-day coastal getaway', template)
        self.assertIn('type="button" class="quick-load"', template)

        # JavaScript binding & safety (single event strategy verification)
        self.assertIn("bindQuickLoadPresets", script)
        self.assertIn("setPrompt", script)
        self.assertIn('byId("userInput").value = text', script)
        self.assertIn('byId("userInput").focus()', script)

        # Ensure no duplicate delegated handling for quick load buttons in document click listener
        self.assertNotIn('document.addEventListener("click", (event) => {\n    const promptButton', script)
        self.assertNotIn('promptButton = event.target.closest', script)

        # Ensure setPrompt only populates input and does NOT auto-submit or call API
        self.assertNotIn("setPrompt() { sendMessage", script)


if __name__ == "__main__":
    unittest.main()
