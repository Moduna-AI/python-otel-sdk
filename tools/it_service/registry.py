"""IT service LangChain tools."""

from langchain.tools import tool

from tools.it_service.schemas import (
    ChangeRequestInput,
    DeviceLookupInput,
    IncidentInput,
    NetworkDiagnosticInput,
    PasswordResetInput,
    RunbookInput,
    ServiceHealthInput,
    SoftwareInstallInput,
    TicketStatusInput,
    UserAccessInput,
)


@tool(args_schema=UserAccessInput)
def check_user_access(user_id: str, system_name: str) -> str:
    """Check a user's current access to a business system."""
    return f"{user_id} has standard user access to {system_name}; privileged access is not enabled."


@tool(args_schema=PasswordResetInput)
def request_password_reset(user_id: str, identity_verified: bool) -> str:
    """Request a password reset after identity verification."""
    if not identity_verified:
        return f"Password reset for {user_id} blocked until identity verification is completed."
    return (
        f"Password reset link generated for {user_id} with a 15-minute expiry."
    )


@tool(args_schema=IncidentInput)
def create_it_incident(summary: str, impact: str) -> str:
    """Create an IT incident with business impact."""
    return f"Incident created: '{summary}' with {impact} impact; assignment group: service desk."


@tool(args_schema=DeviceLookupInput)
def lookup_device_asset(asset_tag: str) -> str:
    """Look up endpoint device assignment and warranty state."""
    return f"Asset {asset_tag}: assigned, encrypted, \
        compliant posture, warranty expires next quarter."


@tool(args_schema=ServiceHealthInput)
def check_service_health(service_name: str) -> str:
    """Check current health for a service or application."""
    return f"{service_name}: degraded latency on one region, no active data-loss incident."


@tool(args_schema=RunbookInput)
def find_it_runbook(issue_type: str) -> str:
    """Find the approved runbook for an IT issue."""
    return f"Runbook for {issue_type}: collect logs, \
        validate scope, apply known workaround, then monitor."


@tool(args_schema=ChangeRequestInput)
def create_change_request(service_name: str, change_summary: str) -> str:
    """Create a standard IT change request."""
    return f"Change request for {service_name} \
        created: {change_summary}; approval required before execution."


@tool(args_schema=SoftwareInstallInput)
def request_software_install(user_id: str, package_name: str) -> str:
    """Request approved software installation for a user."""
    return f"Install request for {package_name} \
        submitted for {user_id}; license check pending."


@tool(args_schema=NetworkDiagnosticInput)
def run_network_diagnostic(hostname: str) -> str:
    """Run a basic network diagnostic summary for a hostname."""
    return f"{hostname}: DNS resolves, ping intermittent, \
        packet loss observed on office Wi-Fi segment."


@tool(args_schema=TicketStatusInput)
def get_it_ticket_status(ticket_id: str) -> str:
    """Get current IT service ticket status."""
    return f"Ticket {ticket_id}: in progress, waiting on endpoint logs, SLA breach risk low."


IT_SERVICE_TOOLS = [
    check_user_access,
    request_password_reset,
    create_it_incident,
    lookup_device_asset,
    check_service_health,
    find_it_runbook,
    create_change_request,
    request_software_install,
    run_network_diagnostic,
    get_it_ticket_status,
]
