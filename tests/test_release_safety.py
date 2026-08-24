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
        self.assertIn("EXPOSE 8080", dockerfile)
        self.assertIn("${PORT:-8080}", dockerfile)
        self.assertIn("fonts-dejavu-core", dockerfile)

    def test_pdf_runtime_dependency_is_pinned_on_its_own_line(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()

        self.assertIn("mcp==1.28.1", requirements)
        self.assertIn("reportlab==5.0.0", requirements)
        self.assertFalse(any("mcp==1.28.1reportlab" in line for line in requirements))

    def test_frontend_contains_hitl_and_sanitized_markdown_paths(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "script.js").read_text(encoding="utf-8")

        self.assertIn("New Trip", template)
        self.assertIn("Approve Itinerary", template)
        self.assertIn("Request Revision", template)
        self.assertIn("dompurify@3.2.6", template)
        self.assertIn("/api/travel/resume", script)
        self.assertIn("/api/travel/download-pdf", script)
        self.assertIn("DOMPurify.sanitize", script)
        self.assertNotIn("html2pdf.bundle.min.js", template)

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

    def test_asset_versioning_and_cache_control(self):
        import os
        from fastapi.testclient import TestClient
        from app import app

        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('/static/style.css?v={{ asset_version }}', template)
        self.assertIn('/static/script.js?v={{ asset_version }}', template)

        client = TestClient(app)

        # 1. Local / default fallback
        if "RENDER_GIT_COMMIT" in os.environ:
            del os.environ["RENDER_GIT_COMMIT"]
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("cache-control"), "no-cache")
        self.assertIn('/static/style.css?v=dev', res.text)
        self.assertIn('/static/script.js?v=dev', res.text)

        # 2. Production RENDER_GIT_COMMIT injection
        os.environ["RENDER_GIT_COMMIT"] = "abcdef123456789xyz"
        try:
            res_prod = client.get("/")
            self.assertEqual(res_prod.status_code, 200)
            self.assertEqual(res_prod.headers.get("cache-control"), "no-cache")
            self.assertIn('/static/style.css?v=abcdef123456', res_prod.text)
            self.assertIn('/static/script.js?v=abcdef123456', res_prod.text)
        finally:
            del os.environ["RENDER_GIT_COMMIT"]


if __name__ == "__main__":
    unittest.main()
