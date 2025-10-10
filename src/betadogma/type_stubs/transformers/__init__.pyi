from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Type
import torch

T = TypeVar('T')

class PreTrainedModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def to(self, device: Any) -> 'PreTrainedModel': ...
    def eval(self) -> 'PreTrainedModel': ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

class PretrainedConfig:
    def __init__(self, **kwargs: Any) -> None: ...
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    max_position_embeddings: int
    type_vocab_size: int
    initializer_range: float
    layer_norm_eps: float
    pad_token_id: int
    position_embedding_type: str
    use_cache: bool
    classifier_dropout: float

class AutoModel:
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str, 
        *args: Any, 
        **kwargs: Any
    ) -> PreTrainedModel: ...

class PreTrainedTokenizer:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: ...
    def encode_plus(
        self,
        text: Union[str, List[str], List[int], List[List[int]]],
        text_pair: Optional[Union[str, List[str], List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, bool, str] = False,
        truncation: Union[bool, str, bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, bool]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]: ...

class AutoTokenizer:
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str, 
        *args: Any, 
        **kwargs: Any
    ) -> PreTrainedTokenizer: ...
    
    def __call__(
        self,
        text: Union[str, List[str], List[int], List[List[int]]],
        text_pair: Optional[Union[str, List[str], List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, bool, str] = False,
        truncation: Union[bool, str, bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, bool]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]: ...
