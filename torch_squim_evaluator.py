import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE


class SquimEvaluator:

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

        enhanced_waveform = self._prepare_audio(enhanced_waveform)

        # Objective metrics
        stoi, pesq, si_sdr = self.obj_model(enhanced_waveform)


        return {
            "stoi": float(stoi.cpu()),
            "pesq": float(pesq.cpu()),
            "si_sdr": float(si_sdr.cpu())
        }
