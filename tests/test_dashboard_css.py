from pathlib import Path


def test_dashboard_cards_use_theme_colors_instead_of_fixed_light_backgrounds() -> None:
    source = Path("src/model_lab/playground/app.py").read_text()

    assert "var(--body-text-color" in source
    assert "var(--background-fill-primary" in source
    assert "var(--background-fill-secondary" in source
    assert "background: #ffffff" not in source
