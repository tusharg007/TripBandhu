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


if __name__ == "__main__":
    unittest.main()
