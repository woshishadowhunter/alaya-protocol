from __future__ import annotations

import unittest
from importlib.metadata import metadata
from pathlib import Path

import alaya


ROOT = Path(__file__).resolve().parents[1]


class PackagingContractTests(unittest.TestCase):
    def test_package_metadata_matches_runtime(self) -> None:
        package_metadata = metadata("alaya-protocol")
        project_urls = dict(
            entry.split(", ", 1) for entry in package_metadata.get_all("Project-URL", [])
        )

        self.assertEqual(package_metadata["Version"], "0.3.1")
        self.assertEqual(package_metadata["Version"], alaya.__version__)
        self.assertEqual(package_metadata["License-Expression"], "Apache-2.0")
        self.assertEqual(package_metadata.get_all("License-File"), ["LICENSE"])
        self.assertNotIn(
            "License :: OSI Approved :: Apache Software License",
            package_metadata.get_all("Classifier", []),
        )
        self.assertEqual(
            project_urls["Repository"],
            "https://github.com/woshishadowhunter/alaya-protocol",
        )

    def test_license_contains_complete_apache_terms(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

        self.assertIn("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION", license_text)
        self.assertIn("9. Accepting Warranty or Additional Liability.", license_text)
        self.assertIn("APPENDIX: How to apply the Apache License to your work.", license_text)

    def test_release_workflow_uses_guarded_trusted_publishing(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")

        self.assertIn("types: [published]", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("id-token: write", workflow)
        self.assertIn("environment: pypi", workflow)
        self.assertIn("vars.PYPI_TRUSTED_PUBLISHING == 'true'", workflow)
        self.assertIn("pypa/gh-action-pypi-publish@release/v1", workflow)

    def test_readme_links_release_wheel_and_integration_guide(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("releases/download/v0.3.1/alaya_protocol-0.3.1-py3-none-any.whl", readme)
        self.assertIn("docs/integrations.md", readme)


if __name__ == "__main__":
    unittest.main()
