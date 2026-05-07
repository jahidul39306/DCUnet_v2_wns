import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE


class SquimEvaluator:
    """
    Evaluate enhanced speech quality using TorchAudio SQUIM models.

    Metrics:
        Objective:
            - STOI
            - PESQ
            - SI-SDR

        Subjective:
            - MOS prediction
    """

    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load pretrained SQUIM models
        self.obj_bundle = SQUIM_OBJECTIVE
        self.subj_bundle = SQUIM_SUBJECTIVE

        self.obj_model = (
            self.obj_bundle.get_model().to(self.device).eval()
        )

        self.subj_model = (
            self.subj_bundle.get_model().to(self.device).eval()
        )

        self.sample_rate = self.obj_bundle.sample_rate

    def _prepare_audio(self, waveform):
        """
        Ensure waveform shape is [1, T] and on correct device.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        return waveform

    @torch.no_grad()
    def evaluate(self, enhanced_waveform):
        """
        Evaluate enhanced speech.

        Args:
            enhanced_waveform (Tensor):
                Shape [T] or [1, T]
                Must match required sample rate.

        Returns:
            dict containing:
                stoi
                pesq
                si_sdr
                mos
        """

        enhanced_waveform = self._prepare_audio(enhanced_waveform)

        # Objective metrics
        stoi, pesq, si_sdr = self.obj_model(enhanced_waveform)

        # Subjective MOS
        # mos = self.subj_model(enhanced_waveform)

        return {
            "stoi": float(stoi.cpu()),
            "pesq": float(pesq.cpu()),
            "si_sdr": float(si_sdr.cpu())
            #"mos": float(mos.cpu()),
        }


# Example usage
if __name__ == "__main__":
    import torchaudio

    audio_path = "enhanced.wav"

    waveform, sr = torchaudio.load(audio_path)

    evaluator = SquimEvaluator()

    if sr != evaluator.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            evaluator.sample_rate,
        )

    scores = evaluator.evaluate(waveform)

    print(scores)