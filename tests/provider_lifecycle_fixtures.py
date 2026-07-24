from __future__ import annotations

from dataclasses import replace

from cluster.aws.qualification import CohortSelectionAuthority
from msctl.aws_hardware import AuthenticatedSelectionBinding
from msctl.aws_lifecycle import (
    AuthenticatedProviderLifecycle,
    ProviderLifecycleBinding,
)


ACCOUNT_ID = "123456789012"
INSTANCE_ID = "i-0123456789abcdef0"
BOOT_ID = "12345678-1234-4abc-8def-1234567890ab"


def provider_lifecycle(profile, *, seed: int = 0) -> AuthenticatedProviderLifecycle:
    dense = AuthenticatedSelectionBinding(
        cohort_id="memorysplit-confirmatory-v3-360m-n10-aws",
        amendment_sha256="1" * 64,
        selection_sha256="2" * 64,
        selection_version_id="selection-version-1",
        profile_id=profile.profile_id,
        provider=profile.provider,
        profile_sha256=profile.sha256,
        runtime_lock_sha256="3" * 64,
        qualification_evidence_sha256="4" * 64,
        environment_receipt_sha256="5" * 64,
        canary_receipt_sha256="6" * 64,
        approval_receipt_sha256="7" * 64,
        approval_public_key_sha256="8" * 64,
        account_id=ACCOUNT_ID,
        instance_id=INSTANCE_ID,
        boot_id=BOOT_ID,
        region="us-east-1",
        availability_zone="us-east-1d",
        purchase_model="on_demand",
        seed=seed,
        arm="dense",
    )
    bindings = {
        "dense": dense,
        "split90": replace(dense, arm="split90"),
    }
    binding = ProviderLifecycleBinding(
        cohort_id=dense.cohort_id,
        provider=dense.provider,
        profile_id=dense.profile_id,
        profile_sha256=dense.profile_sha256,
        hardware_amendment_sha256=dense.amendment_sha256,
        provider_selection_sha256=dense.selection_sha256,
        provider_selection_version_id=dense.selection_version_id,
        runtime_lock_sha256=dense.runtime_lock_sha256,
        runtime_sbom_sha256="9" * 64,
        qualification_evidence_sha256=dense.qualification_evidence_sha256,
        qualification_environment_receipt_sha256=(
            dense.environment_receipt_sha256
        ),
        qualification_canary_receipt_sha256=dense.canary_receipt_sha256,
        qualification_approval_receipt_sha256=dense.approval_receipt_sha256,
        qualification_approval_public_key_sha256=(
            dense.approval_public_key_sha256
        ),
        objective_controls_contract_sha256="a" * 64,
        account_id=dense.account_id,
        instance_id=dense.instance_id,
        boot_id=dense.boot_id,
        region=dense.region,
        availability_zone=dense.availability_zone,
        purchase_model=dense.purchase_model,
        seed=seed,
        arms=("dense", "split90"),
    )
    authority = CohortSelectionAuthority(
        profile=profile,
        seed=seed,
        arms=("dense", "split90"),
        bindings=bindings,
    )
    return AuthenticatedProviderLifecycle(
        binding=binding,
        profile=authority.profile,
        arm_bindings=authority.bindings,
    )
