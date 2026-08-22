def test_package_exposes_version() -> None:
    import heart_disease

    assert heart_disease.__version__ == "2.0.0"
