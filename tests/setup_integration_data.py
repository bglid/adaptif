import configparser
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MS_SNSD = DATA_DIR / "MS-SNSD"
CONFIG = MS_SNSD / "noisyspeech_synthesizer.cfg"


def clone_dataset() -> None:
    """Grab the repo only if it doesn't already exist in the integration test."""
    if MS_SNSD.exists():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/microsoft/MS-SNSD.git",
            MS_SNSD,
        ],
        check=True,
    )

    # NOTE: Microsoft has a bug in their repo. See: https://github.com/microsoft/MS-SNSD/issues/16
    # The fix is essentially to fix the '*_snrlevels' typecast as int
    # See: https://github.com/microsoft/MS-SNSD/pull/11

    # fixes MS bug
    synthesizer = MS_SNSD / "noisyspeech_synthesizer.py"
    text = synthesizer.read_text()
    text = text.replace(
        'total_snrlevels = float(cfg["total_snrlevels"])',
        'total_snrlevels = int(cfg["total_snrlevels"])',
    )
    synthesizer.write_text(text)


def configure_dataset() -> None:
    """Configure the settings of the dataset generator."""
    config = configparser.ConfigParser()
    proof = config.read(CONFIG)
    # debuggin
    print(proof)

    dataset = config["noisy_speech"]

    # editing the params of the dataset
    dataset["audio_length"] = "5"
    dataset["total_hours"] = "0.0015"  # roughly 5 seconds of audio
    dataset["snr_lower"] = "0"
    dataset["snr_upper"] = "20"
    dataset["total_snrlevels"] = "3"

    with CONFIG.open("w") as file:
        config.write(file)


def generate_dataset() -> None:
    """Generate the MS-SNSD dataset given the configured params in `configure_dataset()`"""
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.9",
            "--with-requirements",
            "requirements.txt",
            "noisyspeech_synthesizer.py",
        ],
        cwd=MS_SNSD,
        check=True,
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    clone_dataset()
    configure_dataset()
    generate_dataset()


if __name__ == "__main__":
    main()
