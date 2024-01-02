import logging
import queue
import re
import signal
import sys
from typing import Optional

import numpy as np
import speech_recognition as sr
import torch
import whisper


logger = logging.getLogger("commaai_voice_assistant")

formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


MODEL_NAME = "base.en"

MICROPHONE_SAMPLE_RATE = 16_000

RECOGNIZER_ENERGY_THRESHOLD = 1_500
RECOGNIZER_PHRASE_TIME_LIMIT = 3

COMMAND_WAKE_UP = "hey comma"
COMMAND_PHRASE_TIMEOUT = 4


audio_data: queue.Queue[Optional[sr.AudioData]] = queue.Queue()


def is_wake_up_command(text: str) -> bool:
    return re.sub(r"[^a-z\s]", "", text, flags=re.IGNORECASE).lower() == COMMAND_WAKE_UP


def clean_text(text: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(text))


def signal_handler(*args) -> None:
    logger.debug("Signal arrived")

    audio_data.put(None)


def run() -> None:
    signal.signal(signal.SIGALRM, signal_handler)

    logger.info("🚀 Starting the voice assistant...")

    logger.debug("Loading the model...")

    model = whisper.load_model(MODEL_NAME)

    logger.debug(
        "Available microphone devices: %s", sr.Microphone.list_microphone_names()
    )

    source = sr.Microphone(sample_rate=MICROPHONE_SAMPLE_RATE)

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = RECOGNIZER_ENERGY_THRESHOLD
    recognizer.dynamic_energy_threshold = False

    def callback(_, audio: sr.AudioData):
        signal.alarm(0)

        logger.debug("New audio")

        audio_data.put(audio.get_raw_data())

    with source:
        recognizer.adjust_for_ambient_noise(source)

    logger.info("🎤 Listening in the background...")

    recognizer.listen_in_background(
        source, callback, phrase_time_limit=RECOGNIZER_PHRASE_TIME_LIMIT
    )

    record_text: Optional[list[str]] = None

    while True:
        logger.debug("Waiting to audio...")

        try:
            audio_chunk = audio_data.get()
        except KeyboardInterrupt:
            break

        if audio_chunk is None:
            logger.info("💬 User command: %s", clean_text(record_text))

            record_text = None

            continue

        logger.debug("Converting to text...")

        text = (
            model.transcribe(
                np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0,
                fp16=torch.cuda.is_available(),
            )
            .get("text")
            .strip()
        )

        logger.debug("Text: %s", text)

        if is_wake_up_command(text):
            logger.info("👋 Hello")

            record_text = []

            signal.alarm(COMMAND_PHRASE_TIMEOUT)

        elif record_text is not None:
            record_text.append(text)

            signal.alarm(COMMAND_PHRASE_TIMEOUT)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
