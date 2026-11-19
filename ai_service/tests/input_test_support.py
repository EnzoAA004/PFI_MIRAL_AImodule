"""Helpers for exercising public inference contracts with registered fixtures."""

from pfi_ai_service.input_registry import InputRegistrationRequest, register_server_side_input


def register_fixture_input(monkeypatch, tmp_path, case_id: str, plane: str) -> str:
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "registered-inputs"))
    source_key = "fixture:sagittal_sample" if plane == "sagittal" else "fixture:axial_sample"
    registered = register_server_side_input(
        InputRegistrationRequest(caseId=case_id, plane=plane, sourceKey=source_key)
    )
    return str(registered["inputId"])
