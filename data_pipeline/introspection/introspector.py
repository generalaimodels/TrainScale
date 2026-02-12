# ════════════════════════════════════════════════════════════════════════════════
# Dataset Introspector - Zero-Hardcoding Discovery
# ════════════════════════════════════════════════════════════════════════════════
# Auto-discovers splits, configs, and column schemas from HuggingFace datasets.
# Lazy-loaded, no full dataset materialization required.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

from data_pipeline.core.types import Result, Ok, Err
from data_pipeline.core.errors import (
    IntrospectionError,
    DatasetNotFoundError,
    SplitNotFoundError,
)

if TYPE_CHECKING:
    from datasets import DatasetInfo
    from huggingface_hub import DatasetCardData


# ─────────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────────

# Default split priority (higher index = higher priority)
DEFAULT_SPLIT_PRIORITY: Tuple[str, ...] = (
    "train",
    "training",
    "default",
    "all",
)

# Common config names that indicate "default"
DEFAULT_CONFIG_NAMES: FrozenSet[str] = frozenset({
    "default",
    "plain_text",
    "en",
    "english",
})


# ─────────────────────────────────────────────────────────────────────────────────
# Feature Type Mapping
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FeatureInfo:
    """
    Information about a dataset feature/column.
    
    Attributes:
        name: Column name
        dtype: Data type as string
        is_list: Whether this is a list/sequence type
        is_nested: Whether this contains nested structures
    """
    name: str
    dtype: str
    is_list: bool = False
    is_nested: bool = False


@dataclass(frozen=True, slots=True)
class SplitInfo:
    """
    Information about a dataset split.
    
    Attributes:
        name: Split name
        num_examples: Number of examples (if known)
        num_bytes: Size in bytes (if known)
    """
    name: str
    num_examples: Optional[int] = None
    num_bytes: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ConfigInfo:
    """
    Information about a dataset configuration.
    
    Attributes:
        name: Config name
        splits: Available splits in this config
        description: Config description
    """
    name: str
    splits: Tuple[SplitInfo, ...] = ()
    description: Optional[str] = None


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """
    Complete metadata about a dataset.
    
    Lazy-loaded, immutable, cache-line optimized.
    
    Attributes:
        dataset_id: Full dataset identifier (org/name)
        configs: Available configurations
        default_config: Inferred default config name
        features: Column schema (from first available split)
        description: Dataset description
        license: Dataset license
        tags: Dataset tags
    """
    dataset_id: str
    configs: Tuple[ConfigInfo, ...] = ()
    default_config: Optional[str] = None
    features: Tuple[FeatureInfo, ...] = ()
    description: Optional[str] = None
    license: Optional[str] = None
    tags: Tuple[str, ...] = ()
    
    def get_config(self, name: str) -> Optional[ConfigInfo]:
        """Get config by name."""
        for cfg in self.configs:
            if cfg.name == name:
                return cfg
        return None
    
    def get_all_splits(self, config: Optional[str] = None) -> List[str]:
        """Get all split names, optionally filtered by config."""
        target_config = config or self.default_config
        if target_config:
            cfg = self.get_config(target_config)
            if cfg:
                return [s.name for s in cfg.splits]
        # Return all unique splits across all configs
        splits = set()
        for cfg in self.configs:
            for s in cfg.splits:
                splits.add(s.name)
        return sorted(splits)


# ─────────────────────────────────────────────────────────────────────────────────
# Introspector Class
# ─────────────────────────────────────────────────────────────────────────────────

class DatasetIntrospector:
    """
    Zero-hardcoding discovery of dataset structure.
    
    Uses HuggingFace Hub API to discover:
    - Available configurations
    - Available splits per config
    - Column schema (features)
    - Dataset metadata
    
    Caches results to minimize API calls.
    
    Time Complexity: O(1) per dataset (cached)
    Space Complexity: O(configs * splits * features)
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize introspector.
        
        Args:
            token: Optional HuggingFace API token for private datasets
        """
        self._token = token or os.environ.get("HF_TOKEN")
        self._cache: Dict[str, DatasetMetadata] = {}
    
    def discover(
        self, 
        dataset_id: str,
        trust_remote_code: bool = False,
    ) -> Result[DatasetMetadata, IntrospectionError]:
        """
        Discover dataset structure.
        
        Uses Hub API for metadata, falls back to loading split info
        if detailed config is not available.
        
        Args:
            dataset_id: Dataset identifier (e.g., "tatsu-lab/alpaca")
            trust_remote_code: Whether to trust remote code for loading
            
        Returns:
            Result containing DatasetMetadata or IntrospectionError
        """
        # Check cache
        if dataset_id in self._cache:
            return Ok(self._cache[dataset_id])
        
        try:
            from huggingface_hub import HfApi, hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError
            
            api = HfApi(token=self._token)
            
            # Get dataset info from Hub
            try:
                info = api.dataset_info(dataset_id)
            except Exception as e:
                return Err(DatasetNotFoundError(
                    message=f"Dataset not found: {dataset_id}",
                    dataset_id=dataset_id,
                    cause=e,
                ))
            
            # Extract configs and splits
            configs = self._extract_configs(info)
            default_config = self._infer_default_config(configs)
            
            # Extract features from first available config/split
            features = self._extract_features(dataset_id, default_config)
            
            # Build metadata
            metadata = DatasetMetadata(
                dataset_id=dataset_id,
                configs=tuple(configs),
                default_config=default_config,
                features=tuple(features),
                description=getattr(info, "description", None),
                license=getattr(info, "license", None),
                tags=tuple(info.tags) if info.tags else (),
            )
            
            # Cache result
            self._cache[dataset_id] = metadata
            return Ok(metadata)
            
        except ImportError:
            return Err(IntrospectionError(
                message="huggingface_hub not installed",
                dataset_id=dataset_id,
            ))
        except Exception as e:
            return Err(IntrospectionError(
                message=f"Introspection failed: {e}",
                dataset_id=dataset_id,
                cause=e,
            ))
    
    def _extract_configs(self, info: Any) -> List[ConfigInfo]:
        """Extract config information from dataset info."""
        configs = []
        
        # Try to get configs from card_data
        card_data = getattr(info, "card_data", None)
        if card_data:
            config_list = getattr(card_data, "configs", None)
            if config_list:
                for cfg in config_list:
                    if isinstance(cfg, dict):
                        name = cfg.get("config_name", cfg.get("name", "default"))
                        splits = []
                        for split_info in cfg.get("splits", []):
                            if isinstance(split_info, dict):
                                splits.append(SplitInfo(
                                    name=split_info.get("name", "train"),
                                    num_examples=split_info.get("num_examples"),
                                    num_bytes=split_info.get("num_bytes"),
                                ))
                        configs.append(ConfigInfo(
                            name=name,
                            splits=tuple(splits),
                        ))
        
        # Fallback: try siblings for parquet files structure
        if not configs:
            siblings = getattr(info, "siblings", [])
            configs = self._infer_configs_from_siblings(siblings)
        
        # Fallback: create default config
        if not configs:
            configs.append(ConfigInfo(
                name="default",
                splits=(SplitInfo(name="train"),),
            ))
        
        return configs
    
    def _infer_configs_from_siblings(self, siblings: List[Any]) -> List[ConfigInfo]:
        """Infer configs from file siblings."""
        config_splits: Dict[str, set] = {}
        
        for sibling in siblings:
            rfilename = getattr(sibling, "rfilename", str(sibling))
            # Look for patterns like data/train-00000-of-00001.parquet
            # or config_name/split/file.parquet
            parts = rfilename.split("/")
            
            if len(parts) >= 2:
                # Try to identify config and split
                possible_config = parts[0] if parts[0] not in ("data", ".") else "default"
                for part in parts:
                    if any(s in part.lower() for s in ("train", "test", "valid", "eval")):
                        if possible_config not in config_splits:
                            config_splits[possible_config] = set()
                        # Extract split name
                        for split_name in ("train", "test", "validation", "eval"):
                            if split_name in part.lower():
                                config_splits[possible_config].add(split_name)
                                break
        
        configs = []
        for config_name, splits in config_splits.items():
            configs.append(ConfigInfo(
                name=config_name,
                splits=tuple(SplitInfo(name=s) for s in sorted(splits)),
            ))
        
        return configs
    
    def _infer_default_config(self, configs: List[ConfigInfo]) -> Optional[str]:
        """Infer the default config name."""
        if not configs:
            return None
        
        # Single config is default
        if len(configs) == 1:
            return configs[0].name
        
        # Check for known default names
        for cfg in configs:
            if cfg.name.lower() in DEFAULT_CONFIG_NAMES:
                return cfg.name
        
        # Return first config
        return configs[0].name
    
    def _extract_features(
        self, 
        dataset_id: str, 
        config: Optional[str]
    ) -> List[FeatureInfo]:
        """Extract feature schema from dataset."""
        try:
            from datasets import load_dataset_builder
            
            builder = load_dataset_builder(
                dataset_id,
                name=config,
                token=self._token,
            )
            
            info = builder.info
            if info and info.features:
                return self._parse_features(info.features)
        except Exception:
            pass
        
        return []
    
    def _parse_features(self, features: Any) -> List[FeatureInfo]:
        """Parse HuggingFace Features into FeatureInfo list."""
        result = []
        
        for name, feat in features.items():
            dtype = self._get_feature_dtype(feat)
            is_list = self._is_list_feature(feat)
            is_nested = self._is_nested_feature(feat)
            
            result.append(FeatureInfo(
                name=name,
                dtype=dtype,
                is_list=is_list,
                is_nested=is_nested,
            ))
        
        return result
    
    def _get_feature_dtype(self, feature: Any) -> str:
        """Get dtype string from feature."""
        feat_type = type(feature).__name__
        
        if feat_type == "Value":
            return getattr(feature, "dtype", "string")
        if feat_type == "ClassLabel":
            return "class_label"
        if feat_type in ("Sequence", "list"):
            return "list"
        if feat_type == "Audio":
            return "audio"
        if feat_type == "Image":
            return "image"
        if isinstance(feature, dict):
            return "dict"
        
        return feat_type.lower()
    
    def _is_list_feature(self, feature: Any) -> bool:
        """Check if feature is a list/sequence type."""
        feat_type = type(feature).__name__
        return feat_type in ("Sequence", "list") or "List" in feat_type
    
    def _is_nested_feature(self, feature: Any) -> bool:
        """Check if feature contains nested structures."""
        return isinstance(feature, dict) or type(feature).__name__ == "Features"
    
    def get_default_split(
        self, 
        dataset_id: str,
        config: Optional[str] = None,
    ) -> Result[str, IntrospectionError]:
        """
        Get the default split for a dataset.
        
        Priority: train > training > default > first available
        
        Args:
            dataset_id: Dataset identifier
            config: Optional config name
            
        Returns:
            Result containing split name or IntrospectionError
        """
        result = self.discover(dataset_id)
        if isinstance(result, Err):
            return result
        
        metadata = result.value
        available_splits = metadata.get_all_splits(config)
        
        if not available_splits:
            return Err(SplitNotFoundError(
                message="No splits found",
                dataset_id=dataset_id,
                config=config,
            ))
        
        # Check priority order
        for split in DEFAULT_SPLIT_PRIORITY:
            if split in available_splits:
                return Ok(split)
        
        # Return first available
        return Ok(available_splits[0])
    
    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self._cache.clear()


# ─────────────────────────────────────────────────────────────────────────────────
# Module-level Functions
# ─────────────────────────────────────────────────────────────────────────────────

# Singleton introspector
_introspector: Optional[DatasetIntrospector] = None


def get_introspector(token: Optional[str] = None) -> DatasetIntrospector:
    """Get or create singleton introspector."""
    global _introspector
    if _introspector is None:
        _introspector = DatasetIntrospector(token=token)
    return _introspector


def discover_dataset(
    dataset_id: str,
    token: Optional[str] = None,
) -> Result[DatasetMetadata, IntrospectionError]:
    """
    Discover dataset structure.
    
    Convenience function using singleton introspector.
    
    Args:
        dataset_id: Dataset identifier
        token: Optional HuggingFace token
        
    Returns:
        Result containing DatasetMetadata or error
    """
    introspector = get_introspector(token)
    return introspector.discover(dataset_id)


def get_default_split(
    dataset_id: str,
    config: Optional[str] = None,
    token: Optional[str] = None,
) -> Result[str, IntrospectionError]:
    """
    Get default split for a dataset.
    
    Convenience function using singleton introspector.
    
    Args:
        dataset_id: Dataset identifier
        config: Optional config name
        token: Optional HuggingFace token
        
    Returns:
        Result containing split name or error
    """
    introspector = get_introspector(token)
    return introspector.get_default_split(dataset_id, config)
