# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tensor serialization utilities for cluster communication."""

import pickle
from typing import Any

from distributed_dataloader.cluster.tensor_extractor import TensorRef
from distributed_dataloader.utils.logger import logger


class TensorSerializer:
    """Serialize and deserialize metadata, rebuild data structures.

    This class handles serialization of metadata (data structure with TensorRef placeholders)
    and reconstruction of original data structures by replacing TensorRef with actual tensors.
    """

    @staticmethod
    def serialize_metadata(metadata: dict[str, Any]) -> bytes:
        """Serialize metadata to bytes.

        Args:
            metadata: Metadata dict containing 'structure' and 'tensor_info'.

        Returns:
            Serialized metadata as bytes.
        """
        try:
            serialized = pickle.dumps(metadata)
            logger.debug(f"Serialized metadata: {len(serialized)} bytes")
            return serialized
        except Exception as e:
            logger.error(f"Failed to serialize metadata: {e}")
            raise RuntimeError(f"Metadata serialization failed: {e}") from e

    @staticmethod
    def deserialize_metadata(data: bytes) -> dict[str, Any]:
        """Deserialize metadata from bytes.

        Args:
            data: Serialized metadata bytes.

        Returns:
            Deserialized metadata dict.
        """
        try:
            metadata = pickle.loads(data)
            logger.debug(f"Deserialized metadata: {len(data)} bytes")
            return metadata
        except Exception as e:
            logger.error(f"Failed to deserialize metadata: {e}")
            raise RuntimeError(f"Metadata deserialization failed: {e}") from e

    @staticmethod
    def rebuild(structure: Any, tensors: list[Any]) -> Any:
        """Rebuild original data structure by replacing TensorRef with tensors.

        Args:
            structure: Data structure with TensorRef placeholders.
            tensors: List of tensors to fill in.

        Returns:
            Reconstructed data structure with actual tensors.

        Example:
            >>> structure = {'image': TensorRef(0), 'label': TensorRef(1), 'meta': {'id': 123}}
            >>> tensors = [torch.randn(3, 224, 224), torch.tensor([1])]
            >>> data = TensorSerializer.rebuild(structure, tensors)
            >>> # data = {'image': tensor1, 'label': tensor2, 'meta': {'id': 123}}
        """

        def traverse(obj: Any) -> Any:
            """Recursively rebuild structure."""
            if isinstance(obj, TensorRef):
                # Replace TensorRef with actual tensor
                if obj.index >= len(tensors):
                    raise RuntimeError(f"TensorRef index {obj.index} out of range (tensors length: {len(tensors)})")
                return tensors[obj.index]
            if isinstance(obj, dict):
                # Recursively rebuild dict
                return {k: traverse(v) for k, v in obj.items()}
            if isinstance(obj, list):
                # Recursively rebuild list
                return [traverse(v) for v in obj]
            if isinstance(obj, tuple):
                # Recursively rebuild tuple
                return tuple(traverse(v) for v in obj)
            # Primitive types or other objects, keep as is
            return obj

        result = traverse(structure)
        logger.debug(f"Rebuilt data structure with {len(tensors)} tensors")
        return result

    @staticmethod
    def validate_metadata(metadata: dict[str, Any]) -> bool:
        """Validate metadata structure.

        Args:
            metadata: Metadata dict to validate.

        Returns:
            True if metadata is valid, False otherwise.
        """
        # Validate top-level structure
        if not TensorSerializer._validate_top_level(metadata):
            return False

        # Validate tensor_info entries
        return TensorSerializer._validate_tensor_info_list(metadata["tensor_info"])

    @staticmethod
    def _validate_top_level(metadata: dict[str, Any]) -> bool:
        """Validate top-level metadata structure."""
        if not isinstance(metadata, dict):
            logger.error("Metadata is not a dict")
            return False

        required_fields = ["structure", "tensor_info"]
        for field in required_fields:
            if field not in metadata:
                logger.error(f"Metadata missing '{field}' field")
                return False

        if not isinstance(metadata["tensor_info"], list):
            logger.error("Metadata 'tensor_info' is not a list")
            return False

        return True

    @staticmethod
    def _validate_tensor_info_list(tensor_info_list: list) -> bool:
        """Validate tensor_info list entries."""
        required_fields = ["shape", "dtype", "device", "index"]

        for i, info in enumerate(tensor_info_list):
            if not isinstance(info, dict):
                logger.error(f"tensor_info[{i}] is not a dict")
                return False

            for field in required_fields:
                if field not in info:
                    logger.error(f"tensor_info[{i}] missing '{field}' field")
                    return False

        return True
