"""
Tests for the DualReview tool.
"""

from unittest.mock import Mock, patch

import pytest

from tools.dualreview import DualReviewRequest, DualReviewTool
from tools.models import ToolModelCategory


class TestDualReviewTool:
    """Test suite for DualReviewTool."""

    def test_tool_metadata(self):
        """Test basic tool metadata and configuration."""
        tool = DualReviewTool()

        assert tool.get_name() == "dualreview"
        assert "OpenAI" in tool.get_description()
        assert "Gemini" in tool.get_description()
        assert tool.get_default_temperature() == 1.0  # TEMPERATURE_ANALYTICAL
        assert tool.get_model_category() == ToolModelCategory.EXTENDED_REASONING
        assert tool.requires_model() is False
        assert tool.requires_expert_analysis() is False

    def test_request_validation_step1(self):
        """Test Pydantic request model validation for step 1."""
        request = DualReviewRequest(
            step="Reviewing authentication module for security issues",
            step_number=1,
            total_steps=2,
            next_step_required=True,
            findings="Initial review of auth flow",
            relevant_files=["/src/auth.py", "/src/login.py"],
        )

        assert request.step_number == 1
        assert len(request.relevant_files) == 2
        assert request.review_type == "full"
        assert request.severity_filter == "all"

    def test_request_validation_missing_files_step1(self):
        """Test that step 1 requires relevant_files field."""
        with pytest.raises(ValueError, match="Step 1 requires 'relevant_files'"):
            DualReviewRequest(
                step="Test step",
                step_number=1,
                total_steps=2,
                next_step_required=True,
                findings="Test findings",
                # Missing relevant_files
            )

    def test_request_validation_later_steps(self):
        """Test request validation for steps 2+."""
        request = DualReviewRequest(
            step="Deeper analysis of security concerns",
            step_number=2,
            total_steps=2,
            next_step_required=False,
            findings="Found SQL injection vulnerability in query builder",
            continuation_id="test-id",
        )

        assert request.step_number == 2
        assert request.next_step_required is False

    def test_request_with_review_options(self):
        """Test request with all review-specific options."""
        request = DualReviewRequest(
            step="Security-focused review",
            step_number=1,
            total_steps=2,
            next_step_required=True,
            findings="Initial assessment",
            relevant_files=["/src/api.py"],
            review_type="security",
            focus_on="authentication and authorization",
            severity_filter="high",
        )

        assert request.review_type == "security"
        assert request.focus_on == "authentication and authorization"
        assert request.severity_filter == "high"

    def test_input_schema_generation(self):
        """Test that input schema is generated correctly."""
        tool = DualReviewTool()
        schema = tool.get_input_schema()

        # Core workflow fields should be present
        assert "step" in schema["properties"]
        assert "step_number" in schema["properties"]
        assert "total_steps" in schema["properties"]
        assert "next_step_required" in schema["properties"]
        assert "findings" in schema["properties"]
        assert "relevant_files" in schema["properties"]
        assert "files_checked" in schema["properties"]
        assert "relevant_context" in schema["properties"]
        assert "issues_found" in schema["properties"]

        # Review-specific fields should be present
        assert "review_type" in schema["properties"]
        assert "focus_on" in schema["properties"]
        assert "severity_filter" in schema["properties"]
        assert "images" in schema["properties"]

        # Fields that should NOT be present
        assert "model" not in schema["properties"]
        assert "temperature" not in schema["properties"]
        assert "thinking_mode" not in schema["properties"]
        assert "hypothesis" not in schema["properties"]
        assert "confidence" not in schema["properties"]

    def test_get_required_actions_step1(self):
        """Test required actions for step 1."""
        tool = DualReviewTool()

        actions = tool.get_required_actions(1, "low", "", 2)
        assert len(actions) >= 3
        assert any("Read and understand" in a for a in actions)
        assert any("structure" in a.lower() or "architecture" in a.lower() for a in actions)

    def test_get_required_actions_step2(self):
        """Test required actions for step 2."""
        tool = DualReviewTool()

        actions = tool.get_required_actions(2, "medium", "Found issues", 2)
        assert len(actions) >= 3
        assert any("security" in a.lower() for a in actions)
        assert any("performance" in a.lower() for a in actions)

    def test_get_required_actions_continuation(self):
        """Test required actions for continuation workflows."""
        tool = DualReviewTool()
        mock_request = Mock()
        mock_request.continuation_id = "test-cont-id"

        # Mock the method that extracts continuation_id
        with patch.object(tool, "get_request_continuation_id", return_value="test-cont-id"):
            actions = tool.get_required_actions(1, "low", "", 2, mock_request)
            assert any("Quickly review" in a for a in actions)

    def test_prepare_step_data(self):
        """Test step data preparation."""
        tool = DualReviewTool()
        request = DualReviewRequest(
            step="Test step",
            step_number=1,
            total_steps=2,
            next_step_required=True,
            findings="Test findings",
            relevant_files=["/test.py"],
            files_checked=["/test.py", "/other.py"],
            relevant_context=["TestClass.method"],
            issues_found=[{"severity": "high", "description": "SQL injection"}],
        )

        step_data = tool.prepare_step_data(request)

        assert step_data["step"] == "Test step"
        assert step_data["findings"] == "Test findings"
        assert step_data["relevant_files"] == ["/test.py"]
        assert step_data["files_checked"] == ["/test.py", "/other.py"]
        assert step_data["relevant_context"] == ["TestClass.method"]
        assert len(step_data["issues_found"]) == 1

    def test_should_call_expert_analysis(self):
        """Test that dualreview skips standard expert analysis."""
        tool = DualReviewTool()
        assert tool.should_call_expert_analysis({}) is False
        assert tool.requires_expert_analysis() is False

    def test_build_review_prompt(self):
        """Test review prompt construction."""
        tool = DualReviewTool()
        tool.initial_request = "Review the authentication module"
        tool.consolidated_findings = Mock()
        tool.consolidated_findings.findings = ["Step 1: Found auth issues"]
        tool.consolidated_findings.files_checked = {"/src/auth.py"}
        tool.consolidated_findings.relevant_files = {"/src/auth.py"}
        tool.consolidated_findings.relevant_context = {"AuthService.login"}
        tool.consolidated_findings.issues_found = [{"severity": "high", "description": "Missing input validation"}]

        request = Mock()
        request.step = "Final step"
        request.review_type = "security"
        request.focus_on = "auth flow"
        request.severity_filter = "all"

        prompt = tool._build_review_prompt(request)

        assert "Review the authentication module" in prompt
        assert "security" in prompt
        assert "auth flow" in prompt
        assert "Missing input validation" in prompt
        assert "AuthService.login" in prompt

    def test_build_investigation_summary(self):
        """Test investigation summary construction."""
        tool = DualReviewTool()
        tool.consolidated_findings = Mock()
        tool.consolidated_findings.findings = [
            "Step 1: Reviewed auth module",
            "Step 2: Found SQL injection",
        ]
        tool.consolidated_findings.files_checked = {"/a.py", "/b.py"}
        tool.consolidated_findings.relevant_files = {"/a.py"}
        tool.consolidated_findings.issues_found = [{"severity": "critical"}]

        summary = tool._build_investigation_summary()

        assert "Total steps: 2" in summary
        assert "Files examined: 2" in summary
        assert "Relevant files: 1" in summary
        assert "Issues identified: 1" in summary

    def test_build_final_response(self):
        """Test final response structure."""
        tool = DualReviewTool()
        tool.initial_request = "Review code"
        tool.consolidated_findings = Mock()
        tool.consolidated_findings.findings = ["Step 1: findings"]
        tool.consolidated_findings.files_checked = {"/test.py"}
        tool.consolidated_findings.relevant_files = {"/test.py"}
        tool.consolidated_findings.issues_found = []

        request = Mock()
        request.step_number = 2
        request.total_steps = 2

        reviews = [
            {"provider": "openai", "model": "gpt-5.2", "status": "success", "review": "OpenAI review content"},
            {"provider": "google", "model": "gemini-2.5-pro", "status": "success", "review": "Gemini review content"},
        ]

        response = tool._build_final_response(request, reviews)

        assert response["status"] == "dual_review_complete"
        assert len(response["reviews"]) == 2
        assert response["summary"]["successful_reviews"] == 2
        assert response["summary"]["failed_reviews"] == 0
        assert "openai" in response["summary"]["providers"]
        assert "google" in response["summary"]["providers"]
        assert response["metadata"]["dual_review_complete"] is True

    def test_build_final_response_partial_failure(self):
        """Test final response when one provider fails."""
        tool = DualReviewTool()
        tool.initial_request = "Review code"
        tool.consolidated_findings = Mock()
        tool.consolidated_findings.findings = []
        tool.consolidated_findings.files_checked = set()
        tool.consolidated_findings.relevant_files = set()
        tool.consolidated_findings.issues_found = []

        request = Mock()
        request.step_number = 2
        request.total_steps = 2

        reviews = [
            {"provider": "openai", "model": "gpt-5.2", "status": "success", "review": "OpenAI review"},
            {"provider": "google", "model": "gemini-2.5-pro", "status": "error", "error": "API key invalid"},
        ]

        response = tool._build_final_response(request, reviews)

        assert response["summary"]["successful_reviews"] == 1
        assert response["summary"]["failed_reviews"] == 1

    def test_build_continuation_response(self):
        """Test intermediate step response."""
        tool = DualReviewTool()
        tool.consolidated_findings = Mock()
        tool.consolidated_findings.files_checked = {"/a.py", "/b.py"}
        tool.consolidated_findings.issues_found = [{"severity": "high"}]

        request = DualReviewRequest(
            step="Step 1 analysis",
            step_number=1,
            total_steps=2,
            next_step_required=True,
            findings="Initial findings",
            relevant_files=["/a.py"],
        )

        response = tool._build_continuation_response(request)

        assert response["status"] == "dual_review_in_progress"
        assert response["step_number"] == 1
        assert response["dualreview_status"]["files_examined"] == 2
        assert response["dualreview_status"]["issues_found"] == 1
        assert "MANDATORY" in response["next_steps"]

    @pytest.mark.asyncio
    async def test_call_both_providers_no_providers(self):
        """Test graceful handling when no providers are available."""
        tool = DualReviewTool()

        with patch("providers.registry.ModelProviderRegistry.get_provider", return_value=None):

            request = Mock()
            request.relevant_files = []
            request.images = None

            results = await tool._call_both_providers("test prompt", request)

            assert len(results) == 1
            assert results[0]["status"] == "error"
            assert "No OpenAI or Gemini providers configured" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_call_single_provider(self):
        """Test calling a single provider."""
        from providers.shared.provider_type import ProviderType

        tool = DualReviewTool()

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "Review findings: code looks good"
        mock_provider.generate_content.return_value = mock_response
        mock_provider.get_provider_type.return_value = ProviderType.OPENAI

        config = {
            "provider": mock_provider,
            "provider_type": ProviderType.OPENAI,
            "model_name": "gpt-5.2",
        }

        request = Mock()
        request.relevant_files = []
        request.images = None

        with patch.object(tool, "_prepare_file_content_for_prompt", return_value=("", [])):
            result = await tool._call_single_provider(config, "test prompt", request)

        assert result["status"] == "success"
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-5.2"
        assert "Review findings" in result["review"]

    @pytest.mark.asyncio
    async def test_call_single_provider_with_files(self):
        """Test that file content is included in provider call."""
        from providers.shared.provider_type import ProviderType

        tool = DualReviewTool()

        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "Review with file context"
        mock_provider.generate_content.return_value = mock_response

        config = {
            "provider": mock_provider,
            "provider_type": ProviderType.GOOGLE,
            "model_name": "gemini-2.5-pro",
        }

        request = Mock()
        request.relevant_files = ["/src/auth.py"]
        request.images = None

        with patch.object(tool, "_prepare_file_content_for_prompt", return_value=("def login(): pass", [])):
            result = await tool._call_single_provider(config, "test prompt", request)

        assert result["status"] == "success"
        # Verify generate_content was called with prompt containing file content
        call_args = mock_provider.generate_content.call_args
        assert "CODE FILES" in call_args.kwargs.get("prompt", call_args[1].get("prompt", "")) or "CODE FILES" in str(
            call_args
        )

    def test_store_initial_issue(self):
        """Test initial issue storage."""
        tool = DualReviewTool()
        tool.store_initial_issue("Review the payment module")
        assert tool.initial_request == "Review the payment module"


class TestDualReviewSchema:
    """Test schema generation for DualReview tool."""

    def test_schema_has_review_type_enum(self):
        """Test review_type field has proper enum values."""
        tool = DualReviewTool()
        schema = tool.get_input_schema()

        review_type = schema["properties"]["review_type"]
        assert review_type["type"] == "string"
        assert set(review_type["enum"]) == {"full", "security", "performance", "quick"}

    def test_schema_has_severity_filter_enum(self):
        """Test severity_filter field has proper enum values."""
        tool = DualReviewTool()
        schema = tool.get_input_schema()

        severity = schema["properties"]["severity_filter"]
        assert severity["type"] == "string"
        assert set(severity["enum"]) == {"critical", "high", "medium", "low", "all"}

    def test_schema_excludes_model_field(self):
        """Test that model field is excluded since dualreview picks its own."""
        tool = DualReviewTool()
        schema = tool.get_input_schema()

        assert "model" not in schema["properties"]

    def test_annotations_read_only(self):
        """Test that tool annotations mark it as read-only."""
        tool = DualReviewTool()
        annotations = tool.get_annotations()

        assert annotations is not None
        assert annotations["readOnlyHint"] is True


if __name__ == "__main__":
    import unittest

    unittest.main()
