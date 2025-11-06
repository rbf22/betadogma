import logging
from typing import Optional, cast

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)


class CaduceusEncoder(nn.Module):
    """Wrapper around the Caduceus-PS backbone from the kuleshov-group release."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        use_gradient_checkpointing: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.frozen = freeze
        self.request_checkpointing = use_gradient_checkpointing

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(
            "Initializing Caduceus encoder: model=%s, device=%s, freeze=%s",
            model_name,
            self.device,
            freeze,
        )

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                device_map=None,
            )

            self.model.to(self.device)

            if use_gradient_checkpointing:
                logger.warning(
                    "Caduceus does not implement gradient checkpointing hooks; ignoring request."
                )

            config = getattr(self.model, "config", None)
            if config is not None:
                base_dim = getattr(config, "d_model", None)
                if base_dim is None:
                    base_dim = getattr(config, "hidden_size", None)
                if base_dim is None:
                    logger.warning("Falling back to default hidden size for Caduceus outputs")
                    base_dim = 256
                rcps_active = bool(getattr(config, "rcps", False))
                self.hidden_size = base_dim * 2 if rcps_active else base_dim
                logger.debug(
                    "Caduceus hidden size inferred: base_dim=%s, rcps=%s => hidden_size=%s",
                    base_dim,
                    rcps_active,
                    self.hidden_size,
                )
            else:
                logger.warning("No config found on Caduceus model; defaulting hidden_size=512")
                self.hidden_size = 512

            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()

            with torch.no_grad():
                test_input = torch.zeros(1, 128, dtype=torch.long, device=self.device)
                test_output = self.model(test_input)
                probe = self._extract_hidden(test_output)
                logger.debug("Probe output shape: %s", tuple(probe.shape))
                self.hidden_size = probe.shape[-1]

            logger.info(
                "Successfully loaded Caduceus encoder (%s) with hidden_size=%s",
                model_name,
                self.hidden_size,
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            if torch.cuda.is_available():
                try:
                    logger.error(
                        "GPU memory at failure: allocated=%.2f GB reserved=%.2f GB",
                        torch.cuda.memory_allocated() / 1e9,
                        torch.cuda.memory_reserved() / 1e9,
                    )
                except Exception:  # pragma: no cover
                    logger.exception("Unable to fetch CUDA memory stats after failure")
            raise RuntimeError(f"Failed to initialize Caduceus encoder: {exc}") from exc

    @staticmethod
    def _extract_hidden(outputs: torch.Tensor | tuple | object) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "last_hidden_state"):
            return cast(torch.Tensor, outputs.last_hidden_state)
        if isinstance(outputs, tuple) and outputs:
            candidate = outputs[0]
            if isinstance(candidate, torch.Tensor):
                return candidate
        raise ValueError(f"Unexpected output type from Caduceus model: {type(outputs)}")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> SimpleNamespace:
        if self.frozen:
            self.model.eval()

        input_ids = input_ids.to(self.device)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        if attention_mask is not None and attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad() if self.frozen else torch.enable_grad():
            outputs = self.model(input_ids=input_ids)
            hidden = self._extract_hidden(outputs)
        return SimpleNamespace(last_hidden_state=hidden)


class SimpleNamespace:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state

