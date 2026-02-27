"""
DualReview tool - Parallel code review from OpenAI and Gemini simultaneously

This tool provides a structured code review workflow that automatically sends code
to both OpenAI and Gemini providers for independent review, then presents both
perspectives side-by-side. It eliminates the need to manually request opinions
from multiple models.

Key features:
- Step-by-step code review workflow (like codereview)
- Automatic parallel review from both OpenAI and Gemini on completion
- Independent, blinded reviews (neither model sees the other's output)
- Side-by-side presentation of both reviews
- Falls back gracefully if only one provider is available
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from mcp.types import TextContent

from config import TEMPERATURE_ANALYTICAL
from systemprompts.dualreview_prompt import DUALREVIEW_PROMPT
from tools.shared.base_models import ConsolidatedFindings, WorkflowRequest
from utils.conversation_memory import MAX_CONVERSATION_TURNS, create_thread, get_thread

from .workflow.base import WorkflowTool

logger = logging.getLogger(__name__)

# Tool-specific field descriptions
DUALREVIEW_FIELD_DESCRIPTIONS = {
    "step": (
        "Review narrative. Step 1: outline the review strategy and initial findings. "
        "Step 2: report deeper findings. The tool sends code to both OpenAI and Gemini "
        "when the review completes. Reference code via `relevant_files`; avoid dumping large snippets."
    ),
    "step_number": "Current review step (starts at 1). Each step should build on the last.",
    "total_steps": (
        "Number of review steps planned. Default: 2 steps (investigation + summary). "
        "Use the same limits when continuing an existing review via continuation_id."
    ),
    "next_step_required": (
        "True when another review step follows. Step 1 -> True, final step -> False. "
        "When False, the tool triggers parallel review from both OpenAI and Gemini."
    ),
    "findings": "Capture findings (positive and negative) across quality, security, performance, and architecture.",
    "files_checked": "Absolute paths of every file reviewed, including those ruled out.",
    "relevant_files": "Step 1: list all files/dirs under review. Must be absolute full non-abbreviated paths.",
    "relevant_context": "Functions or methods central to findings (e.g. 'Class.method' or 'function_name').",
    "issues_found": "Issues with severity (critical/high/medium/low) and descriptions.",
    "images": "Optional diagram or screenshot paths that clarify review context.",
    "review_type": "Review focus: full, security, performance, or quick.",
    "focus_on": "Optional note on areas to emphasise (e.g. 'threading', 'auth flow').",
    "severity_filter": "Lowest severity to include when reporting issues (critical/high/medium/low/all).",
}


class DualReviewRequest(WorkflowRequest):
    """Request model for dual review workflow steps"""

    # Required fields for each step
    step: str = Field(..., description=DUALREVIEW_FIELD_DESCRIPTIONS["step"])
    step_number: int = Field(..., description=DUALREVIEW_FIELD_DESCRIPTIONS["step_number"])
    total_steps: int = Field(..., description=DUALREVIEW_FIELD_DESCRIPTIONS["total_steps"])
    next_step_required: bool = Field(..., description=DUALREVIEW_FIELD_DESCRIPTIONS["next_step_required"])

    # Investigation tracking fields
    findings: str = Field(..., description=DUALREVIEW_FIELD_DESCRIPTIONS["findings"])
    files_checked: list[str] = Field(default_factory=list, description=DUALREVIEW_FIELD_DESCRIPTIONS["files_checked"])
    relevant_files: list[str] = Field(default_factory=list, description=DUALREVIEW_FIELD_DESCRIPTIONS["relevant_files"])
    relevant_context: list[str] = Field(
        default_factory=list, description=DUALREVIEW_FIELD_DESCRIPTIONS["relevant_context"]
    )
    issues_found: list[dict] = Field(default_factory=list, description=DUALREVIEW_FIELD_DESCRIPTIONS["issues_found"])

    # Deprecated confidence field kept for compatibility
    confidence: str | None = Field("low", exclude=True)

    # Optional images
    images: list[str] | None = Field(default=None, description=DUALREVIEW_FIELD_DESCRIPTIONS["images"])

    # Review-specific fields (used in step 1)
    review_type: Literal["full", "security", "performance", "quick"] | None = Field(
        "full", description=DUALREVIEW_FIELD_DESCRIPTIONS["review_type"]
    )
    focus_on: str | None = Field(None, description=DUALREVIEW_FIELD_DESCRIPTIONS["focus_on"])
    severity_filter: Literal["critical", "high", "medium", "low", "all"] | None = Field(
        "all", description=DUALREVIEW_FIELD_DESCRIPTIONS["severity_filter"]
    )

    # Override inherited fields to exclude from schema
    temperature: float | None = Field(default=None, exclude=True)
    thinking_mode: str | None = Field(default=None, exclude=True)

    # Not used in dualreview
    hypothesis: str | None = Field(None, exclude=True)

    @model_validator(mode="after")
    def validate_step_one_requirements(self):
        """Ensure step 1 has required relevant_files field."""
        if self.step_number == 1 and not self.relevant_files:
            raise ValueError("Step 1 requires 'relevant_files' field to specify code files or directories to review")
        return self


class DualReviewTool(WorkflowTool):
    """
    Dual Review tool for parallel code review from OpenAI and Gemini.

    This tool implements a structured code review workflow where the CLI agent
    investigates code through systematic steps, then automatically sends the
    code and findings to both OpenAI and Gemini for independent expert review.
    Both reviews are returned side-by-side.
    """

    def __init__(self):
        super().__init__()
        self.initial_request: str | None = None
        self.review_config: dict = {}

    def get_name(self) -> str:
        return "dualreview"

    def get_description(self) -> str:
        return (
            "Parallel code review from both OpenAI and Gemini simultaneously. "
            "Use when you want independent expert opinions from multiple AI providers "
            "without specifying models manually. Guides through structured investigation, "
            "then sends code to both providers for independent review."
        )

    def get_system_prompt(self) -> str:
        return DUALREVIEW_PROMPT

    def get_default_temperature(self) -> float:
        return TEMPERATURE_ANALYTICAL

    def get_model_category(self) -> ToolModelCategory:
        """Dual review requires extended reasoning."""
        from tools.models import ToolModelCategory

        return ToolModelCategory.EXTENDED_REASONING

    def get_workflow_request_model(self):
        """Return the dual review workflow-specific request model."""
        return DualReviewRequest

    def requires_model(self) -> bool:
        """DualReview manages its own model selection — no model needed at MCP boundary."""
        return False

    def requires_expert_analysis(self) -> bool:
        """DualReview handles its own dual-model calls instead of standard expert analysis."""
        return False

    def get_input_schema(self) -> dict[str, Any]:
        """Generate input schema for dual review workflow."""
        from .workflow.schema_builders import WorkflowSchemaBuilder

        field_overrides = {
            "step": {
                "type": "string",
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["step"],
            },
            "step_number": {
                "type": "integer",
                "minimum": 1,
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["step_number"],
            },
            "total_steps": {
                "type": "integer",
                "minimum": 1,
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["total_steps"],
            },
            "next_step_required": {
                "type": "boolean",
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["next_step_required"],
            },
            "findings": {
                "type": "string",
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["findings"],
            },
            "files_checked": {
                "type": "array",
                "items": {"type": "string"},
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["files_checked"],
            },
            "relevant_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["relevant_files"],
            },
            "relevant_context": {
                "type": "array",
                "items": {"type": "string"},
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["relevant_context"],
            },
            "issues_found": {
                "type": "array",
                "items": {"type": "object"},
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["issues_found"],
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["images"],
            },
            "review_type": {
                "type": "string",
                "enum": ["full", "security", "performance", "quick"],
                "default": "full",
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["review_type"],
            },
            "focus_on": {
                "type": "string",
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["focus_on"],
            },
            "severity_filter": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low", "all"],
                "default": "all",
                "description": DUALREVIEW_FIELD_DESCRIPTIONS["severity_filter"],
            },
        }

        # Exclude fields not used in dualreview
        excluded_workflow_fields = [
            "hypothesis",
            "confidence",
        ]

        excluded_common_fields = [
            "model",  # DualReview picks its own models
            "temperature",
            "thinking_mode",
        ]

        return WorkflowSchemaBuilder.build_schema(
            tool_specific_fields=field_overrides,
            model_field_schema=None,
            auto_mode=False,
            tool_name=self.get_name(),
            excluded_workflow_fields=excluded_workflow_fields,
            excluded_common_fields=excluded_common_fields,
            require_model=False,
        )

    def get_required_actions(
        self, step_number: int, confidence: str, findings: str, total_steps: int, request=None
    ) -> list[str]:
        """Define required actions for each review phase."""
        if request:
            continuation_id = self.get_request_continuation_id(request)
            if continuation_id:
                if step_number == 1:
                    return [
                        "Quickly review the code files to understand context",
                        "Identify any critical issues that need immediate attention",
                        "Prepare summary of key findings for dual-provider review",
                    ]
                else:
                    return ["Complete review and proceed to dual-provider analysis"]

        if step_number == 1:
            return [
                "Read and understand the code files specified for review",
                "Examine the overall structure, architecture, and design patterns used",
                "Identify the main components, classes, and functions in the codebase",
                "Look for obvious issues: bugs, security concerns, performance problems",
                "Note any code smells, anti-patterns, or areas of concern",
            ]
        elif step_number >= 2:
            return [
                "Examine specific code sections you've identified as concerning",
                "Analyze security implications: input validation, authentication, authorization",
                "Check for performance issues: algorithmic complexity, resource usage",
                "Look for architectural problems: tight coupling, missing abstractions",
                "Identify code quality issues: readability, maintainability, error handling",
            ]
        else:
            return [
                "Continue examining the codebase for additional patterns and potential issues",
                "Focus on areas that haven't been thoroughly examined yet",
            ]

    def should_call_expert_analysis(self, consolidated_findings, request=None) -> bool:
        """DualReview handles its own model calls — skip standard expert analysis."""
        return False

    def prepare_expert_analysis_context(self, consolidated_findings) -> str:
        """Not used — dualreview builds its own prompt for each provider."""
        return ""

    def prepare_step_data(self, request) -> dict:
        """Map dual review fields for internal processing."""
        return {
            "step": request.step,
            "step_number": request.step_number,
            "findings": request.findings,
            "files_checked": request.files_checked,
            "relevant_files": request.relevant_files,
            "relevant_context": request.relevant_context,
            "issues_found": request.issues_found,
            "hypothesis": request.findings,
            "images": request.images or [],
            "confidence": "high",
        }

    def store_initial_issue(self, step_description: str):
        """Store initial request for review context."""
        self.initial_request = step_description

    async def execute_workflow(self, arguments: dict[str, Any]) -> list:
        """Override execute_workflow to handle dual-provider review on completion."""
        request = self.get_workflow_request_model()(**arguments)

        # Resolve or create continuation thread
        continuation_id = request.continuation_id

        if request.step_number == 1:
            if not continuation_id:
                clean_args = {k: v for k, v in arguments.items() if k not in ["_model_context", "_resolved_model_name"]}
                continuation_id = create_thread(self.get_name(), clean_args)
                request.continuation_id = continuation_id
                arguments["continuation_id"] = continuation_id
                self.work_history = []
                self.consolidated_findings = ConsolidatedFindings()

            self.store_initial_issue(request.step)
            self.initial_request = request.step
            self.review_config = {
                "relevant_files": request.relevant_files,
                "review_type": request.review_type,
                "focus_on": request.focus_on,
                "severity_filter": request.severity_filter,
            }

        # Track workflow state
        step_data = self.prepare_step_data(request)
        self.work_history.append(step_data)
        self._update_consolidated_findings(step_data)

        # If more steps needed, return guidance for continued investigation
        if request.next_step_required:
            response_data = self._build_continuation_response(request)

            if continuation_id:
                self.store_conversation_turn(continuation_id, response_data, request)
                offer = self._build_continuation_offer_data(continuation_id)
                if offer:
                    response_data["continuation_offer"] = offer

            return [TextContent(type="text", text=json.dumps(response_data, indent=2, ensure_ascii=False))]

        # Final step — send to both providers in parallel
        review_prompt = self._build_review_prompt(request)
        reviews = await self._call_both_providers(review_prompt, request)

        response_data = self._build_final_response(request, reviews)

        if continuation_id:
            self.store_conversation_turn(continuation_id, response_data, request)
            offer = self._build_continuation_offer_data(continuation_id)
            if offer:
                response_data["continuation_offer"] = offer

        return [TextContent(type="text", text=json.dumps(response_data, indent=2, ensure_ascii=False))]

    def _build_continuation_response(self, request) -> dict:
        """Build response for intermediate workflow steps."""
        required_actions = self.get_required_actions(
            request.step_number, "medium", request.findings, request.total_steps, request
        )

        return {
            "status": "dual_review_in_progress",
            "step_number": request.step_number,
            "total_steps": request.total_steps,
            "dualreview_status": {
                "steps_completed": request.step_number,
                "files_examined": len(self.consolidated_findings.files_checked),
                "issues_found": len(self.consolidated_findings.issues_found),
            },
            "next_steps": (
                "MANDATORY: DO NOT call dualreview again immediately. You MUST first examine "
                "the code files thoroughly. REQUIRED ACTIONS:\n"
                + "\n".join(f"{i + 1}. {action}" for i, action in enumerate(required_actions))
                + f"\n\nOnly call dualreview again AFTER completing your investigation. "
                f"Use step_number: {request.step_number + 1} and report specific findings."
            ),
            "metadata": {
                "tool_name": self.get_name(),
                "workflow_type": "dual_provider_review",
            },
        }

    def _build_review_prompt(self, request) -> str:
        """Build the review prompt sent to both providers."""
        parts = []

        # Review request context
        parts.append(f"=== CODE REVIEW REQUEST ===\n{self.initial_request or request.step}\n=== END REQUEST ===")

        # Review configuration
        config_parts = []
        if request.review_type:
            config_parts.append(f"- Review type: {request.review_type}")
        if request.focus_on:
            config_parts.append(f"- Focus areas: {request.focus_on}")
        if request.severity_filter:
            config_parts.append(f"- Severity filter: {request.severity_filter}")
        if config_parts:
            parts.append("\n=== REVIEW CONFIGURATION ===\n" + "\n".join(config_parts) + "\n=== END CONFIGURATION ===")

        # Investigation summary
        summary = self._build_investigation_summary()
        parts.append(f"\n=== AGENT'S INVESTIGATION SUMMARY ===\n{summary}\n=== END INVESTIGATION ===")

        # Issues found during investigation
        if self.consolidated_findings.issues_found:
            issues_text = "\n".join(
                f"[{issue.get('severity', 'unknown').upper()}] {issue.get('description', 'No description')}"
                for issue in self.consolidated_findings.issues_found
            )
            parts.append(f"\n=== ISSUES IDENTIFIED BY AGENT ===\n{issues_text}\n=== END ISSUES ===")

        # Relevant code elements
        if self.consolidated_findings.relevant_context:
            methods_text = "\n".join(f"- {method}" for method in self.consolidated_findings.relevant_context)
            parts.append(f"\n=== RELEVANT CODE ELEMENTS ===\n{methods_text}\n=== END CODE ELEMENTS ===")

        return "\n".join(parts)

    def _build_investigation_summary(self) -> str:
        """Build summary of the investigation steps."""
        summary_parts = [
            f"Total steps: {len(self.consolidated_findings.findings)}",
            f"Files examined: {len(self.consolidated_findings.files_checked)}",
            f"Relevant files: {len(self.consolidated_findings.relevant_files)}",
            f"Issues identified: {len(self.consolidated_findings.issues_found)}",
            "",
        ]

        for finding in self.consolidated_findings.findings:
            summary_parts.append(finding)

        return "\n".join(summary_parts)

    async def _call_both_providers(self, review_prompt: str, request) -> list[dict]:
        """Call both OpenAI and Gemini in parallel for independent code review."""
        from providers.registry import ModelProviderRegistry
        from providers.shared.provider_type import ProviderType
        from tools.models import ToolModelCategory

        # Resolve best model for each provider
        provider_configs = []

        for provider_type in [ProviderType.OPENAI, ProviderType.GOOGLE]:
            provider = ModelProviderRegistry.get_provider(provider_type)
            if not provider:
                logger.info(f"[DUALREVIEW] Provider {provider_type.value} not available, skipping")
                continue

            # Get provider's preferred model for code review
            allowed_models = ModelProviderRegistry._get_allowed_models_for_provider(provider, provider_type)
            if not allowed_models:
                logger.info(f"[DUALREVIEW] No allowed models for {provider_type.value}, skipping")
                continue

            preferred_model = provider.get_preferred_model(ToolModelCategory.EXTENDED_REASONING, list(allowed_models))
            if not preferred_model:
                preferred_model = sorted(allowed_models)[0]

            provider_configs.append(
                {
                    "provider": provider,
                    "provider_type": provider_type,
                    "model_name": preferred_model,
                }
            )

        if not provider_configs:
            logger.warning("[DUALREVIEW] No providers available for dual review")
            return [
                {
                    "provider": "none",
                    "model": "none",
                    "status": "error",
                    "error": "No OpenAI or Gemini providers configured. Check your API keys.",
                }
            ]

        # Call all available providers in parallel
        tasks = [self._call_single_provider(config, review_prompt, request) for config in provider_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        reviews = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                config = provider_configs[i]
                logger.exception(f"[DUALREVIEW] Error from {config['provider_type'].value}: {result}")
                reviews.append(
                    {
                        "provider": config["provider_type"].value,
                        "model": config["model_name"],
                        "status": "error",
                        "error": str(result),
                    }
                )
            else:
                reviews.append(result)

        return reviews

    async def _call_single_provider(self, config: dict, review_prompt: str, request) -> dict:
        """Call a single provider for code review."""
        from utils.model_context import ModelContext

        provider = config["provider"]
        model_name = config["model_name"]
        provider_type = config["provider_type"]

        model_context = ModelContext(model_name=model_name)

        # Add file content to the prompt
        prompt = review_prompt
        if request.relevant_files:
            file_content, _ = self._prepare_file_content_for_prompt(
                request.relevant_files,
                None,  # Blinded — no conversation history
                "Code files for review",
                model_context=model_context,
            )
            if file_content:
                prompt = f"{prompt}\n\n=== CODE FILES ===\n{file_content}\n=== END CODE FILES ==="

        system_prompt = self.get_system_prompt()

        # Validate temperature
        validated_temperature, temp_warnings = self.validate_and_correct_temperature(
            self.get_default_temperature(), model_context
        )
        for warning in temp_warnings:
            logger.warning(warning)

        response = provider.generate_content(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=validated_temperature,
            thinking_mode="high",
            images=request.images if request.images else None,
        )

        return {
            "provider": provider_type.value,
            "model": model_name,
            "status": "success",
            "review": response.content,
            "metadata": {
                "provider": provider_type.value,
                "model_name": model_name,
            },
        }

    def _build_final_response(self, request, reviews: list[dict]) -> dict:
        """Build the final response combining both provider reviews."""
        successful_reviews = [r for r in reviews if r["status"] == "success"]
        failed_reviews = [r for r in reviews if r["status"] == "error"]

        response_data = {
            "status": "dual_review_complete",
            "step_number": request.step_number,
            "total_steps": request.total_steps,
            "reviews": reviews,
            "summary": {
                "providers_consulted": len(reviews),
                "successful_reviews": len(successful_reviews),
                "failed_reviews": len(failed_reviews),
                "providers": [r.get("provider", "unknown") for r in reviews],
                "models_used": [r.get("model", "unknown") for r in reviews],
            },
            "investigation": {
                "initial_request": self.initial_request,
                "steps_completed": len(self.consolidated_findings.findings),
                "files_examined": list(self.consolidated_findings.files_checked),
                "relevant_files": list(self.consolidated_findings.relevant_files),
                "issues_found_by_agent": self.consolidated_findings.issues_found,
            },
            "next_steps": (
                "DUAL REVIEW IS COMPLETE. You MUST now:\n"
                "1. Present BOTH reviews side-by-side to the user\n"
                "2. Highlight where both reviewers AGREE (high-confidence issues)\n"
                "3. Highlight where they DISAGREE (needs user judgment)\n"
                "4. Synthesize a unified priority list of issues\n"
                "5. Provide concrete, actionable recommendations\n"
                "6. Categorize all issues by severity (Critical > High > Medium > Low)"
            ),
            "metadata": {
                "tool_name": self.get_name(),
                "workflow_type": "dual_provider_review",
                "models_consulted": [r.get("model", "unknown") for r in reviews],
                "dual_review_complete": True,
            },
        }

        return response_data

    def _build_continuation_offer_data(self, continuation_id: str) -> dict | None:
        """Create continuation offer data."""
        try:
            from tools.models import ContinuationOffer

            thread = get_thread(continuation_id)
            if thread and thread.turns:
                remaining_turns = max(0, MAX_CONVERSATION_TURNS - len(thread.turns))
            else:
                remaining_turns = MAX_CONVERSATION_TURNS - 1

            note = (
                f"Dual review workflow can continue for {remaining_turns} more exchanges."
                if remaining_turns > 0
                else "Dual review workflow continuation limit reached."
            )

            return ContinuationOffer(
                continuation_id=continuation_id,
                note=note,
                remaining_turns=remaining_turns,
            ).model_dump()
        except Exception:
            return None

    def customize_workflow_response(self, response_data: dict, request) -> dict:
        """Customize response for dual review workflow."""
        if request.step_number == 1:
            self.initial_request = request.step
            if request.relevant_files:
                self.review_config = {
                    "relevant_files": request.relevant_files,
                    "review_type": request.review_type,
                    "focus_on": request.focus_on,
                    "severity_filter": request.severity_filter,
                }
        return response_data

    # Required abstract methods from BaseTool

    def get_request_model(self):
        """Return the dual review workflow-specific request model."""
        return DualReviewRequest

    async def prepare_prompt(self, request) -> str:
        """Not used — workflow tools use execute_workflow()."""
        return ""
