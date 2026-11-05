# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenPI transforms for the three-camera ARX/Aloha-format Jimu policy."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.policies import aloha_policy
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override


@dataclasses.dataclass(frozen=True)
class LeRobotArxDataConfig(DataConfigFactory):
    """Matches the transforms used by the 77-source arm-fixed checkpoint."""

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.base_0",
                            "cam_left_wrist": "observation.images.left_wrist_0",
                            "cam_right_wrist": "observation.images.right_wrist_0",
                        },
                        "state": "observation.state",
                        "tactile": "observation.tactile",
                        "tactile_mask": "observation.tactile_mask",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                aloha_policy.AlohaInputs(adapt_to_pi=False),
                _transforms.DirectResizeImages(224, 224),
            ],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=False)],
        )
        if not getattr(model_config, "use_tactile", False):
            raise ValueError("LeRobotArxDataConfig requires use_tactile=True")
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
        )
