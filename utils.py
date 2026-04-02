import logging
import tempfile
import soundfile as sf
import atexit
import shutil

# remove temporary wave files
_TEMP_DIR = tempfile.mkdtemp()
atexit.register(shutil.rmtree, _TEMP_DIR, True) 



def set_up_logging(output_file_path):
    log_file = output_file_path.replace(".jsonl", ".log")

    # Clear existing handlers if rerunning in interactive environments
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set a unified format without milliseconds
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to root logger
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    logging.getLogger("unbabel_comet").setLevel(logging.WARNING)

    for name in [
        "lightning",
        "lightning_fabric",
        "pytorch_lightning",
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=".*LeafSpec.*",
    )

    warnings.filterwarnings(
        "ignore",
        message=".*srun.*",
    )

def audio_to_tempfile(audio_dict):
    if audio_dict is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", dir=_TEMP_DIR, delete=False)
    sf.write(tmp.name, audio_dict["array"], audio_dict["sampling_rate"])
    return tmp.name