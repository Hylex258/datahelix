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
"""Tensor extraction utilities for cluster communication."""

from typing import Any

from distributed_dataloader.utils.adapt_frameworks import get_backend
from distributed_dataloader.utils.logger import logger


class TensorRef:
    """Placeholder for tensor in data structure.

    This class is used to replace tensors in the data structure during extraction.
    It stores the index of the tensor in the extracted tensor list.
    """

    def __init__(self, index: int):
        """Initialize TensorRef.

        Args:
            index: Index of the tensor in the extracted tensor list.
        """
        self.index = index

    def __repr__(self) -> str:
        return f"TensorRef({self.index})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, TensorRef):
            return False
        return self.index == other.index

    def __hash__(self) -> int:
        return hash(self.index)


class TensorExtractor:
    """Extract tensors from complex data structures.

    This class recursively traverses data structures (dict, list, tuple) and extracts
    all tensors, replacing them with TensorRef placeholders. The extracted tensors
    and metadata can be used for efficient transmission via communication primitives.
    """

    @staticmethod
    def extract(data: Any) -> tuple[list[Any], dict[str, Any]]:
        """Extract tensors from data structure.

        Args:
            data: Input data structure (can be dict, list, tuple, tensor, or primitive types).

        Returns:
            A tuple of (tensors, metadata) where:
            - tensors: List of extracted tensors
            - metadata: Dict containing:
                - 'structure': Data structure with TensorRef placeholders
                - 'tensor_info': List of tensor metadata (shape, dtype, device)

        Example:
            >>> data = {'image': torch.randn(3, 224, 224), 'label': torch.tensor([1]), 'meta': {'id': 123}}
            >>> tensors, metadata = TensorExtractor.extract(data)
            >>> # tensors = [tensor1, tensor2]
            >>> # metadata = {
            >>> #     'structure': {'image': TensorRef(0), 'label': TensorRef(1), 'meta': {'id': 123}},
            >>> #     'tensor_info': [
            >>> #         {'shape': (3, 224, 224), 'dtype': 'torch.float32', 'device': 'cpu', 'index': 0},
            >>> #         {'shape': (1,), 'dtype': 'torch.int64', 'device': 'cpu', 'index': 1}
            >>> #     ]
            >>> # }
        """
        tensors = []
        framework = get_backend()

        def traverse(obj: Any) -> Any:
            """Recursively traverse and extract tensors."""
            if TensorExtractor._is_tensor(obj, framework):
                # Extract tensor
                idx = len(tensors)
                tensors.append(obj)
                return TensorRef(idx)
            if isinstance(obj, dict):
                # Recursively process dict
                return {k: traverse(v) for k, v in obj.items()}
            if isinstance(obj, list):
                # Recursively process list
                return [traverse(v) for v in obj]
            if isinstance(obj, tuple):
                # Recursively process tuple
                return tuple(traverse(v) for v in obj)
            # Primitive types or other objects, keep as is
            return obj

        # Traverse and extract
        structure = traverse(data)

        # Build tensor info
        tensor_info = []
        for idx, tensor in enumerate(tensors):
            info = TensorExtractor._get_tensor_info(tensor, framework, idx)
            tensor_info.append(info)

        metadata = {
            "structure": structure,
            "tensor_info": tensor_info,
        }

        logger.debug(f"Extracted {len(tensors)} tensors from data structure")

        return tensors, metadata

    @staticmethod
    def _is_tensor(obj: Any, framework: str) -> bool:
        """Check if object is a tensor.

        Args:
            obj: Object to check.
            framework: Framework name ('PyTorch' or 'MindSpore').

        Returns:
            True if object is a tensor, False otherwise.
        """
        if framework == "PyTorch":
            try:
                import torch

                return isinstance(obj, torch.Tensor)
            except ImportError:
                return False
        elif framework == "MindSpore":
            try:
                import mindspore

                return isinstance(obj, mindspore.Tensor)
            except ImportError:
                return False
        else:
            return False

    @staticmethod
    def _get_tensor_info(tensor: Any, framework: str, index: int) -> dict[str, Any]:
        """Get metadata information from tensor.

        Args:
            tensor: Tensor object.
            framework: Framework name ('PyTorch' or 'MindSpore').
            index: Index of tensor in the list.

        Returns:
            Dict containing tensor metadata.
        """
        if framework == "PyTorch":
            return {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "index": index,
            }
        if framework == "MindSpore":
            return {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "index": index,
            }
        raise RuntimeError(f"Unsupported framework: {framework}")

    @staticmethod
    def contains_tensor(data: Any) -> bool:
        """Check if data structure contains any tensor.

        Args:
            data: Input data structure.

        Returns:
            True if data contains at least one tensor, False otherwise.
        """
        framework = get_backend()

        def traverse(obj: Any) -> bool:
            """Recursively check for tensors."""
            if TensorExtractor._is_tensor(obj, framework):
                return True
            if isinstance(obj, dict):
                return any(traverse(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return any(traverse(v) for v in obj)
            return False

        return traverse(data)
