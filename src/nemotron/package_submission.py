import zipfile

from dotenv import load_dotenv

from nemotron.config import NemotronConfig


def main() -> None:
    load_dotenv(override=True)
    cfg = NemotronConfig.from_env()

    if not cfg.output_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {cfg.output_dir}. Run training first."
        )

    cfg.submission_zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(cfg.submission_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(cfg.output_dir.iterdir()):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)

    with zipfile.ZipFile(cfg.submission_zip_path, "r") as zf:
        names = zf.namelist()
        if "adapter_config.json" not in names:
            raise RuntimeError(
                "Generated zip is invalid for submission: missing adapter_config.json"
            )

    size_mb = cfg.submission_zip_path.stat().st_size / (1024 * 1024)
    print(f"Created {cfg.submission_zip_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
