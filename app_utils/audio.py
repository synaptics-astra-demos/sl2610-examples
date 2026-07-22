import subprocess
import logging
import os
import time
import numpy as np

logger = logging.getLogger(__name__)


class AudioManager:
    def __init__(self, device=None, sample_rate=16000, channels=2):
        self._astra_version = self._get_astra_version()
        self._device = device or self._get_usb_audio_device()
        self._sample_rate = sample_rate
        self.arecord_process = None
        self._channels = channels
        
        # If Astra SDK 1.6 and specific headset detected → force 48kHz
        if self._astra_version == "1.6.0":
            try:
                aplay_output = subprocess.check_output("aplay -l", shell=True, text=True)
                if any(name in aplay_output for name in ["H3 [INZONE H3]", "SPACE [SPACE]"]):
                    self._sample_rate = 48000
                    self._channels = 1
                    logger.info("Assigned 48000 Hz sample rate for device on Astra SDK v%s", self._astra_version)
            except Exception as e:
                logger.warning("Failed to parse aplay output for headset detection: %s", str(e))

    @property
    def device(self):
        """Get the current audio device."""
        return self._device

    @device.setter
    def device(self, new_device):
        """Set a new audio device."""
        self._device = new_device

    @property
    def sample_rate(self):
        """Get the current sample rate."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, new_sample_rate):
        """Set a new sample rate."""
        self._sample_rate = new_sample_rate

    def play(self, filename):
        """Play the audio file using the specified audio device."""
        if not self._device:
            raise RuntimeError("Audio device not set.")
        
        device = "plugplay" if getattr(self, "_astra_version", "") == "1.6.0" else self._device
        logger.debug(f"Playing through: {device}")
        subprocess.run(["aplay", "-q", "-D", device, filename], check=True)

    def start_arecord(self, chunk_size=512):
        """Start the arecord subprocess."""
        if not self._device:
            raise RuntimeError("Audio device not set.")
        if self.arecord_process:
            self.stop_arecord()
        command = f"arecord -D {self._device} -f S16_LE -r {self._sample_rate} -c {self._channels}"
        self.arecord_process = subprocess.Popen(
            #command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=chunk_size, shell=True
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )

    def stop_arecord(self):
        """Stop the arecord subprocess."""
        if self.arecord_process:
            self.arecord_process.terminate()
            self.arecord_process.wait()
            self.arecord_process = None

    def read(self, chunk_size=512):
        """Read audio data from the arecord subprocess."""
        if not self.arecord_process:
            raise RuntimeError("arecord process not running.")

        while True:
            data = self.arecord_process.stdout.read(chunk_size * 4)
            if not data:
                break
            yield np.frombuffer(data, dtype=np.int16)[::2].astype(np.float32) / 32768.0

    def wait_for_audio(self, timeout: float = 5.0):
        """Wait until a USB audio device is available."""
        logger.debug('Waiting for audio device...')
        start = time.time()
        while True:
            process = os.popen("aplay -l | grep USB\\ Audio && sleep 0.5")
            output = process.read()
            process.close()
            if 'USB Audio' in output:
                logger.info("Found USB audio device: %s", output.strip("\n"))
                return True
            if time.time() - start > timeout:
                logger.warning(
                    "Timed out after %.1fs waiting for a USB audio device. "
                    "Ensure the device is connected and recognized by the system (check `aplay -l`).",
                    timeout
                )
                return False
            time.sleep(0.5) # avoid busy-loop

    def _get_astra_version(self):
        """Return Astra SDK version if available."""
        try:
            with open("/etc/astra_version", "r") as f:
                return f.read().strip()
        except Exception:
            return ""

    def _create_asoundrc_for_sdk_1_6(self):
        """Write ~/.asoundrc configuration for Astra SDK v1.6 using detected USB Audio device."""
        try:
            pcm_output = subprocess.check_output("cat /proc/asound/pcm", shell=True, text=True)
            for line in pcm_output.splitlines():
                line_lower = line.lower()
                # Check for USB audio device with playback and capture capabilities
                # Example line: "00-01: USB Audio : USB Audio : playback 1 : capture 1"
                if "usb" in line_lower and "audio" in line_lower and "playback" in line_lower and "capture" in line_lower:
                    hw_id = line.split(":")[0].strip()
                    card, dev = [str(int(x)) for x in hw_id.split("-")]
                    name = line.split(":")[1].strip().split()[0]
                    asoundrc_content = f"""
pcm.plugplay {{
    type plug
    slave {{
        pcm "hw:{card},{dev}"
        period_size 1024
        buffer_size 2048
    }}
}}
"""
                    with open(os.path.expanduser("~/.asoundrc"), "w") as f:
                        f.write(asoundrc_content)
                    logger.debug(f"~/.asoundrc created using USB Audio card {card}, device {dev} ({name})")
                    return
            logger.warning("No USB audio device found in /proc/asound/pcm")
        except Exception as e:
            logger.error(f"Failed to create ~/.asoundrc: {e}")

    def _get_usb_audio_device(self):
        """Finds the audio device ID for a USB Audio device using `aplay -l`."""
        if not self.wait_for_audio():
            return None

        if self._astra_version == "1.6.0":
            self._create_asoundrc_for_sdk_1_6()
            logger.info("Using 'plugplay' device for Astra SDK v1.6.0")
            return "plugplay"

        try:
            result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            for line in lines:
                if "USB Audio" in line:
                    card_line = line.split()
                    card_index = card_line[1][:-1]  # Removes trailing colon
                    device_name = f"plughw:{card_index},0"
                    # print(f"Found audio device: {device_name}")
                    return device_name
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running `aplay -l`: {e}")
            return None

        return "default"
