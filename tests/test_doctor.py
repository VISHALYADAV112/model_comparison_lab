from model_lab import doctor


def test_doctor_checks_the_real_sam_image_entrypoint(monkeypatch) -> None:
    imported: list[str] = []

    def record_import(module: str):
        imported.append(module)
        return object()

    monkeypatch.setattr(doctor.importlib, "import_module", record_import)

    ready, errors = doctor._package_report()

    assert ready["official_sam3"] is True
    assert errors == {}
    assert "sam3.model.sam3_image_processor" in imported
    assert imported.count("sam3.model.sam3_image_processor") == 1
