"""Pydantic schemas for IT service tools."""

from pydantic import BaseModel, Field


class UserAccessInput(BaseModel):
    """Input for access checks."""

    user_id: str = Field(description="Employee or contractor identifier.")
    system_name: str = Field(description="Business system name.")


class PasswordResetInput(BaseModel):
    """Input for password reset requests."""

    user_id: str = Field(description="Employee or contractor identifier.")
    identity_verified: bool = Field(
        description="Whether identity verification has passed."
    )


class IncidentInput(BaseModel):
    """Input for incident creation."""

    summary: str = Field(description="Incident summary.")
    impact: str = Field(
        description="Impact level such as low, medium, high, or critical."
    )


class DeviceLookupInput(BaseModel):
    """Input for device lookup."""

    asset_tag: str = Field(description="Device asset tag.")


class ServiceHealthInput(BaseModel):
    """Input for service health checks."""

    service_name: str = Field(description="Service or application name.")


class RunbookInput(BaseModel):
    """Input for runbook lookup."""

    issue_type: str = Field(description="Issue category or alert name.")


class ChangeRequestInput(BaseModel):
    """Input for change requests."""

    service_name: str = Field(description="Service affected by the change.")
    change_summary: str = Field(
        description="Short description of the proposed change."
    )


class SoftwareInstallInput(BaseModel):
    """Input for software installation requests."""

    user_id: str = Field(description="Employee or contractor identifier.")
    package_name: str = Field(description="Software package name.")


class NetworkDiagnosticInput(BaseModel):
    """Input for network diagnostics."""

    hostname: str = Field(description="Hostname or device name to diagnose.")


class TicketStatusInput(BaseModel):
    """Input for IT ticket status."""

    ticket_id: str = Field(description="IT service ticket identifier.")
