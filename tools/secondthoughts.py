"""
Second Thoughts tool - Get independent expert feedback from OpenAI and Gemini simultaneously

This tool provides a structured workflow that automatically sends your request to
both OpenAI and Gemini providers for independent expert feedback, then presents both
perspectives side-by-side. Works for code reviews, implementation plan feedback,
architecture assessments, and technical decision evaluation.

Key features:
- Step-by-step investigation workflow
- Automatic parallel feedback from both OpenAI and Gemini on completion
- Independent, blinded reviews (neither model sees the other's output)
- Side-by-side presentation of both expert opinions
- Falls back gracefully if only one provider is available
- Flexible modes: code review, plan evaluation, architecture, security, performance
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
from systemprompts.secondthoughts_prompt import SECONDTHOUGHTS_PROMPT
from tools.shared.base_models import ConsolidatedFindings, WorkflowRequest
from utils.conversation_memory import MAX_CONVERSATION_TURNS, create_thread, get_thread

from .workflow.base import WorkflowTool

logger = logging.getLogger(__name__)

# Tool-specific field descriptions
SECONDTHOUGHTS_FIELD_DESCRIPTIONS = {
    "step": (
        "Your analysis narrative. Step 1: outline what you're asking about and initial observations. "
        "Step 2: report deeper findings. The tool sends everything to both OpenAI and Gemini "
        "when the workflow completes. Use `relevant_files` for code; put plans/ideas in `findings`."
    ),
    "step_number": "Current step (starts at 1). Each step should build on the last.",
    "total_steps": (
        "Number of steps planned. Default: 2 steps (investigation + summary). "
        "Use the same limits when continuing an existing workflow via continuation_id."
    ),
    "next_step_required": (
        "True when another step follows. Step 1 -> True, final step -> False. "
        "When False, the tool sends everything to both OpenAI and Gemini for independent feedback."
    ),
    "findings": (
        "Your observations, the plan/proposal being evaluated, or context for the question. "
        "For plan reviews, include the full plan here. For code reviews, summarize findings."
    ),
    "files_checked": "Absolute paths of every file examined, including those ruled out.",
    "relevant_files": "Files/dirs relevant to the request. Must be absolute full non-abbreviated paths. Optional for plan/architecture mode.",
    "relevant_context": "Functions, methods, or components central to the discussion (e.g. 'Class.method' or 'auth subsystem').",
    "issues_found": "Issues or concerns with severity (critical/high/medium/low) and descriptions.",
    "images": "Optional diagram or screenshot paths that clarify context.",
    "mode": (
        "What kind of feedback you want: 'review' (code review), 'plan' (implementation plan feedback), "
        "'architecture' (design/architecture assessment), 'security' (security-focused), "
        "'performance' (performance-focused), or 'general' (open-ended expert opinion)."
    ),
    "focus_on": "Optional note on areas to emphasise (e.g. 'threading', 'auth flow', 'migration risk', 'scalability').",
    "severity_filter": "Lowest severity to include when reporting issues (critical/high/medium/low/all).",
}


class SecondThoughtsRequest(WorkflowRequest):
    """Request model for second thoughts workflow steps"""

    # Required fields for each step
    step: str = Field(..., description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["step"])
    step_number: int = Field(..., description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["step_number"])
    total_steps: int = Field(..., description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["total_steps"])
    next_step_required: bool = Field(..., description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["next_step_required"])

    # Investigation tracking fields
    findings: str = Field(..., description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["findings"])
    files_checked: list[str] = Field(
        default_factory=list, description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["files_checked"]
    )
    relevant_files: list[str] = Field(
        default_factory=list, description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["relevant_files"]
    )
    relevant_context: list[str] = Field(
        default_factory=list, description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["relevant_context"]
    )
    issues_found: list[dict] = Field(
        default_factory=list, description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["issues_found"]
    )

    # Deprecated confidence field kept for compatibility
    confidence: str | None = Field("low", exclude=True)

    # Optional images
    images: list[str] | None = Field(default=None, description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["images"])

    # Mode and focus fields (used in step 1)
    mode: Literal["review", "plan", "architecture", "security", "performance", "general"] | None = Field(
        "review", description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["mode"]
    )
    focus_on: str | None = Field(None, description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["focus_on"])
    severity_filter: Literal["critical", "high", "medium", "low", "all"] | None = Field(
        "all", description=SECONDTHOUGHTS_FIELD_DESCRIPTIONS["severity_filter"]
    )

    # Override inherited fields to exclude from schema
    temperature: float | None = Field(default=None, exclude=True)
    thinking_mode: str | None = Field(default=None, exclude=True)

    # Not used in secondthoughts
    hypothesis: str | None = Field(None, exclude=True)

    @model_validator(mode="after")
    def validate_step_one_requirements(self):
        """Ensure step 1 has either relevant_files or substantive findings."""
        if self.step_number == 1 and not self.relevant_files and not self.findings:
            raise ValueError(
                "Step 1 requires either 'relevant_files' (for code) or 'findings' "
                "(for plans/proposals) to give the experts something to review"
            )
        return self


class SecondThoughtsTool(WorkflowTool):
    """
    Second Thoughts — get independent expert feedback from OpenAI and Gemini simultaneously.

    Works for code reviews, implementation plan feedback, architecture assessments,
    and general technical decision evaluation. The CLI agent investigates through
    structured steps, then automatically sends the context to both providers for
    independent expert feedback. Both opinions are returned side-by-side.
    """

    def __init__(self):
        super().__init__()
        self.initial_request: str | None = None
        self.review_config: dict = {}

    def get_name(self) -> str:
        return "secondthoughts"

    def get_description(self) -> str:
        return (
            "Get a second opinion from both OpenAI and Gemini simultaneously. "
            "Use for code reviews, implementation plan feedback, architecture assessments, "
            "or any technical question where you want independent expert opinions from "
            "multiple AI providers. Guides through structured investigation, then sends "
            "context to both providers for independent feedback."
        )

    def get_system_prompt(self) -> str:
        return SECONDTHOUGHTS_PROMPT

    def get_default_temperature(self) -> float:
        return TEMPERATURE_ANALYTICAL

    def get_model_category(self) -> ToolModelCategory:
        """Second thoughts requires extended reasoning."""
        from tools.models import ToolModelCategory

        return ToolModelCategory.EXTENDED_REASONING

    def get_workflow_request_model(self):
        """Return the second thoughts workflow-specific request model."""
        return SecondThoughtsRequest

    def requires_model(self) -> bool:
        """SecondThoughts manages its own model selection — no model needed at MCP boundary."""
        return False

    def requires_expert_analysis(self) -> bool:
        """SecondThoughts handles its own dual-model calls instead of standard expert analysis."""
        return False

    def get_input_schema(self) -> dict[str, Any]:
        """Generate input schema for second thoughts workflow."""
        from .workflow.schema_builders import WorkflowSchemaBuilder

        field_overrides = {
            "step": {
                "type": "string",
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["step"],
            },
            "step_number": {
                "type": "integer",
                "minimum": 1,
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["step_number"],
            },
            "total_steps": {
                "type": "integer",
                "minimum": 1,
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["total_steps"],
            },
            "next_step_required": {
                "type": "boolean",
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["next_step_required"],
            },
            "findings": {
                "type": "string",
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["findings"],
            },
            "files_checked": {
                "type": "array",
                "items": {"type": "string"},
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["files_checked"],
            },
            "relevant_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["relevant_files"],
            },
            "relevant_context": {
                "type": "array",
                "items": {"type": "string"},
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["relevant_context"],
            },
            "issues_found": {
                "type": "array",
                "items": {"type": "object"},
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["issues_found"],
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["images"],
            },
            "mode": {
                "type": "string",
                "enum": ["review", "plan", "architecture", "security", "performance", "general"],
                "default": "review",
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["mode"],
            },
            "focus_on": {
                "type": "string",
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["focus_on"],
            },
            "severity_filter": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low", "all"],
                "default": "all",
                "description": SECONDTHOUGHTS_FIELD_DESCRIPTIONS["severity_filter"],
            },
        }

        # Exclude fields not used in secondthoughts
        excluded_workflow_fields = [
            "hypothesis",
            "confidence",
        ]

        excluded_common_fields = [
            "model",  # SecondThoughts picks its own models
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
        """Define required actions for each investigation phase."""
        if request:
            continuation_id = self.get_request_continuation_id(request)
            if continuation_id:
                if step_number == 1:
                    return [
                        "Quickly review the provided context to understand what feedback is needed",
                        "Identify any critical concerns that need immediate attention",
                        "Prepare summary of key observations for dual-provider feedback",
                    ]
                else:
                    return ["Complete investigation and proceed to dual-provider analysis"]

        mode = getattr(request, "mode", "review") if request else "review"

        if mode in ("plan", "architecture", "general"):
            if step_number == 1:
                return [
                    "Understand the proposal, plan, or question being presented",
                    "Identify the stated goals, constraints, and assumptions",
                    "Note any gaps, risks, or unstated assumptions",
                    "Consider alternative approaches or simplifications",
                    "If files are referenced, read them to understand the existing codebase context",
                ]
            else:
                return [
                    "Evaluate feasibility and complexity of the proposed approach",
                    "Assess trade-offs: complexity vs. value, short-term vs. long-term",
                    "Check for missing considerations: error handling, migration, rollback, edge cases",
                    "Consider operational concerns: monitoring, debugging, maintenance burden",
                ]
        else:
            # Code review / security / performance modes
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
        """SecondThoughts handles its own model calls — skip standard expert analysis."""
        return False

    def prepare_expert_analysis_context(self, consolidated_findings) -> str:
        """Not used — secondthoughts builds its own prompt for each provider."""
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
                "mode": request.mode,
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
            "status": "second_thoughts_in_progress",
            "step_number": request.step_number,
            "total_steps": request.total_steps,
            "secondthoughts_status": {
                "steps_completed": request.step_number,
                "files_examined": len(self.consolidated_findings.files_checked),
                "issues_found": len(self.consolidated_findings.issues_found),
            },
            "next_steps": (
                "MANDATORY: DO NOT call secondthoughts again immediately. You MUST first "
                "investigate the context thoroughly. REQUIRED ACTIONS:\n"
                + "\n".join(f"{i + 1}. {action}" for i, action in enumerate(required_actions))
                + f"\n\nOnly call secondthoughts again AFTER completing your investigation. "
                f"Use step_number: {request.step_number + 1} and report specific findings."
            ),
            "metadata": {
                "tool_name": self.get_name(),
                "workflow_type": "dual_provider_review",
            },
        }

    def _build_review_prompt(self, request) -> str:
        """Build the prompt sent to both providers."""
        parts = []

        # Request context
        mode = getattr(request, "mode", "review") or "review"
        mode_labels = {
            "review": "CODE REVIEW",
            "plan": "IMPLEMENTATION PLAN REVIEW",
            "architecture": "ARCHITECTURE ASSESSMENT",
            "security": "SECURITY REVIEW",
            "performance": "PERFORMANCE REVIEW",
            "general": "EXPERT FEEDBACK",
        }
        label = mode_labels.get(mode, "EXPERT FEEDBACK")
        parts.append(f"=== {label} REQUEST ===\n{self.initial_request or request.step}\n=== END REQUEST ===")

        # Configuration
        config_parts = []
        if mode:
            config_parts.append(f"- Mode: {mode}")
        if request.focus_on:
            config_parts.append(f"- Focus areas: {request.focus_on}")
        if request.severity_filter:
            config_parts.append(f"- Severity filter: {request.severity_filter}")
        if config_parts:
            parts.append("\n=== CONFIGURATION ===\n" + "\n".join(config_parts) + "\n=== END CONFIGURATION ===")

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
                logger.info(f"[SECONDTHOUGHTS] Provider {provider_type.value} not available, skipping")
                continue

            # Get provider's preferred model for code review
            allowed_models = ModelProviderRegistry._get_allowed_models_for_provider(provider, provider_type)
            if not allowed_models:
                logger.info(f"[SECONDTHOUGHTS] No allowed models for {provider_type.value}, skipping")
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
            logger.warning("[SECONDTHOUGHTS] No providers available for dual review")
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
                logger.exception(f"[SECONDTHOUGHTS] Error from {config['provider_type'].value}: {result}")
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
        """Call a single provider for expert feedback."""
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
                "Relevant files",
                model_context=model_context,
            )
            if file_content:
                prompt = f"{prompt}\n\n=== RELEVANT FILES ===\n{file_content}\n=== END RELEVANT FILES ==="

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
            "status": "second_thoughts_complete",
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
                "SECOND THOUGHTS IS COMPLETE. You MUST now:\n"
                "1. Present BOTH expert opinions side-by-side to the user\n"
                "2. Highlight where both experts AGREE (high-confidence findings)\n"
                "3. Highlight where they DISAGREE (needs user judgment)\n"
                "4. Synthesize a unified priority list of recommendations\n"
                "5. Provide concrete, actionable next steps"
            ),
            "metadata": {
                "tool_name": self.get_name(),
                "workflow_type": "dual_provider_review",
                "models_consulted": [r.get("model", "unknown") for r in reviews],
                "second_thoughts_complete": True,
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
                f"Second thoughts workflow can continue for {remaining_turns} more exchanges."
                if remaining_turns > 0
                else "Second thoughts workflow continuation limit reached."
            )

            return ContinuationOffer(
                continuation_id=continuation_id,
                note=note,
                remaining_turns=remaining_turns,
            ).model_dump()
        except Exception:
            return None

    def customize_workflow_response(self, response_data: dict, request) -> dict:
        """Customize response for second thoughts workflow."""
        if request.step_number == 1:
            self.initial_request = request.step
            self.review_config = {
                "relevant_files": request.relevant_files,
                "mode": request.mode,
                "focus_on": request.focus_on,
                "severity_filter": request.severity_filter,
            }
        return response_data

    # Required abstract methods from BaseTool

    def get_request_model(self):
        """Return the second thoughts workflow-specific request model."""
        return SecondThoughtsRequest

    async def prepare_prompt(self, request) -> str:
        """Not used — workflow tools use execute_workflow()."""
        return ""
